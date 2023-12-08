#include <sstream>
#include <filesystem>
#include <tuple>
#include <map>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "pybind11/stl/filesystem.h"
#include "BinauralSpatializer/3DTI_BinauralSpatializer.h"
#include "HRTF/HRTFCereal.h"
#include "HRTF/HRTFFactory.h"
#include "ILD/ILDCereal.h"
#include "BRIR/BRIRCereal.h"
#include "BRIR/BRIRFactory.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)


namespace py = pybind11;
using namespace pybind11::literals;
using namespace Common;
using namespace Binaural;

typedef std::tuple<float,float,float> Position;
typedef std::tuple<float,float,float,float> Orientation;
typedef std::vector<Position> Positions;
typedef std::vector<Orientation> Orientations;
typedef std::map<const std::shared_ptr<CSingleSourceDSP>, const py::array_t<float>> SourceSamplesMap;
typedef std::map<const std::shared_ptr<CSingleSourceDSP>, const Position> SourcePositionMap;
typedef std::map<const std::shared_ptr<CSingleSourceDSP>, const Positions> SourcePositionsMap;

void updateTransform(CTransform &transform, const std::optional<Position>& position, const std::optional<Orientation>& orientation = std::nullopt)
{
    if (position) {
        transform.SetPosition(CVector3(std::get<0>(*position), std::get<1>(*position), std::get<2>(*position)));
    }
    if (orientation) {
        transform.SetOrientation(CQuaternion(std::get<0>(*orientation), std::get<1>(*orientation), std::get<2>(*orientation), std::get<3>(*orientation)));
    }
}

void updateListenerPositionAndOrientation(const std::shared_ptr<CListener>& listener, const std::optional<Position>& position, const std::optional<Orientation>& orientation)
{
    if (position || orientation) {
        CTransform transform = listener->GetListenerTransform();
        updateTransform(transform, position, orientation);
        listener->SetListenerTransform(transform);
    }
}

void updateListenerPositionAndOrientation(const std::shared_ptr<CListener>& listener, const size_t blockIdx, const Positions& positions, const Orientations& orientations) {
    const std::optional<const Position> position = (blockIdx < positions.size()) ? std::optional<const Position>(positions[blockIdx]) : std::nullopt;
    const std::optional<const Orientation> orientation = (blockIdx < orientations.size()) ? std::optional<const Orientation>(orientations[blockIdx]) : std::nullopt;
    updateListenerPositionAndOrientation(listener, position, orientation);
}

void updateSourcePosition(const std::shared_ptr<CSingleSourceDSP>& source, const std::optional<Position>& position)
{
    if (position) {
        CTransform transform = source->GetCurrentSourceTransform();
        updateTransform(transform, position);
        source->SetSourceTransform(transform);
    }
}

void updateSourcePosition(const std::shared_ptr<CSingleSourceDSP>& source, const SourcePositionMap &positionMap)
{
    const auto position = positionMap.find(source) != positionMap.end() ? std::optional<const Position>(positionMap.find(source)->second) : std::nullopt;
    updateSourcePosition(source, position);
}

void updateSourcePosition(const std::shared_ptr<CSingleSourceDSP>& source, const size_t blockIdx, const SourcePositionsMap& positionsMap)
{
    std::optional<Position> position;
    if (positionsMap.find(source) != positionsMap.end()) {
        const Positions positions = positionsMap.find(source)->second;
        if (blockIdx < positions.size()) {
            position = std::optional<const Position>(positions[blockIdx]);
        }
    }
    updateSourcePosition(source, position);
}

class BinauralStreamer
{
public:
    BinauralStreamer(CCore binauralRenderer)
    : m_binauralRenderer(binauralRenderer)
    , m_bufferSize(binauralRenderer.GetAudioState().bufferSize)
    , m_inputBuffer(m_bufferSize)
    , m_leftBuffer(m_bufferSize)
    , m_rightBuffer(m_bufferSize)
    , m_start(0)
    {
    }

protected:
    void processSourceSamples(const std::shared_ptr<CSingleSourceDSP>& source, const py::array_t<float>& samples, float* const leftPtr, float* const rightPtr, const py::ssize_t sourceSize, const py::ssize_t sourceEnd)
    {
        const auto samplesMemory = samples.unchecked<1>();
        std::copy(samplesMemory.data(m_start), samplesMemory.data(sourceEnd), m_inputBuffer.begin());
        std::fill(m_inputBuffer.begin()+sourceSize, m_inputBuffer.end(), 0.f);
        source->SetBuffer(m_inputBuffer);
        source->ProcessAnechoic(m_leftBuffer, m_rightBuffer);
        addToOutput(sourceSize, leftPtr, rightPtr);
    }

    void processEnvironments(const py::ssize_t size, float* const leftPtr, float* const rightPtr) {
        for (const auto& environment : m_binauralRenderer.GetEnvironments()) {
            environment->ProcessVirtualAmbisonicReverb(m_leftBuffer, m_rightBuffer);
            addToOutput(size, leftPtr, rightPtr);
        }
    }

    void addToOutput(const py::ssize_t size, float* const leftPtr, float* const rightPtr)
    {
        std::transform(m_leftBuffer.begin(), m_leftBuffer.begin()+size, leftPtr, leftPtr, std::plus<float>());
        std::transform(m_rightBuffer.begin(), m_rightBuffer.begin()+size, rightPtr, rightPtr, std::plus<float>());
    }

    const CCore m_binauralRenderer;
    const int m_bufferSize;
    CMonoBuffer<float> m_inputBuffer;
    CMonoBuffer<float> m_leftBuffer;
    CMonoBuffer<float> m_rightBuffer;
    py::ssize_t m_start;
};

class FiniteBinauralStreamer: public BinauralStreamer
{
public:
    FiniteBinauralStreamer(CCore binauralRenderer, const SourceSamplesMap& samplesMap)
    : BinauralStreamer(binauralRenderer)
    , m_samplesMap(samplesMap)
    {
        std::vector<py::ssize_t> sourceLengths;
        for (const auto& kv : samplesMap) {
            sourceLengths.push_back(kv.second.size());
        }
        m_binauralLength = *std::max_element(sourceLengths.begin(), sourceLengths.end());
    }

    size_t size() const
    {
        return std::ceil(static_cast<double>(m_binauralLength) / m_bufferSize);
    }

    py::array_t<float, py::array::f_style> operator()(const SourcePositionMap& positionMap, const std::optional<const Position>& listenerPosition = std::nullopt, const std::optional<const Orientation>& listenerOrientation = std::nullopt)
    {
        if (m_start >= m_binauralLength) {
            throw py::stop_iteration("All source samples have been processed.");
        }
        py::array_t<float, py::array::f_style> binauralSamples({static_cast<py::ssize_t>(m_bufferSize), py::ssize_t(2)});
        binauralSamples[py::ellipsis()] = 0.f;
        auto binauralMem = binauralSamples.mutable_unchecked<2>();
        // Update listener position and orientation if given
        updateListenerPositionAndOrientation(m_binauralRenderer.GetListener(), listenerPosition, listenerOrientation);
        // Update sources
        const py::ssize_t nextStart = m_start + m_bufferSize;
        for (const auto& [source, samples] : m_samplesMap) {
            // Update source position if given
            updateSourcePosition(source, positionMap);
            // Process source samples if any still left
            if (m_start < samples.size()) {
                const py::ssize_t sourceEnd = std::min(nextStart, samples.size());
                const py::ssize_t sourceSize = sourceEnd - m_start;
                processSourceSamples(source, samples, binauralMem.mutable_data(0, 0), binauralMem.mutable_data(0, 1), sourceSize, sourceEnd);
            }
        }
        // Update environments
        processEnvironments(m_bufferSize, binauralMem.mutable_data(0, 0), binauralMem.mutable_data(0, 1));
        m_start += m_bufferSize;
        return binauralSamples;
    }

protected:
    const SourceSamplesMap m_samplesMap;
    py::ssize_t m_binauralLength;
};

class OfflineFiniteBinauralStreamer: public FiniteBinauralStreamer
{
public:
    OfflineFiniteBinauralStreamer(CCore binauralRenderer, const SourceSamplesMap& samplesMap, const SourcePositionsMap& positionsMap = SourcePositionsMap(), const Positions& listenerPositions = Positions(), const Orientations& listenerOrientations = Orientations())
    : FiniteBinauralStreamer(binauralRenderer, samplesMap)
    , m_binauralSamples({m_binauralLength, py::ssize_t(2)})
    {
        m_binauralSamples[py::ellipsis()] = 0.f;
        auto binauralMem = m_binauralSamples.mutable_unchecked<2>();
        for (size_t blockIdx = 0; m_start < m_binauralLength; m_start += m_bufferSize, ++blockIdx) {
            // Update listener position and orientation if given
            updateListenerPositionAndOrientation(m_binauralRenderer.GetListener(), blockIdx, listenerPositions, listenerOrientations);
            // Update sources
            const py::ssize_t blockEnd = std::min(m_start + m_bufferSize, m_binauralLength);
            for (const auto& [source, samples] : samplesMap) {
                // Update source position if given
                updateSourcePosition(source, blockIdx, positionsMap);
                // Process source samples if any still left
                if (m_start < samples.size()) {
                    const py::ssize_t sourceEnd = std::min(blockEnd, samples.size());
                    const py::ssize_t sourceSize = sourceEnd - m_start;
                    processSourceSamples(source, samples, binauralMem.mutable_data(m_start, 0), binauralMem.mutable_data(m_start, 1), sourceSize, sourceEnd);
                }
            }
            // Update environments
            const py::ssize_t blockSize = blockEnd - m_start;
            processEnvironments(blockSize, binauralMem.mutable_data(m_start, 0), binauralMem.mutable_data(m_start, 1));
        }
    }

    const py::array_t<float, py::array::f_style>& operator()()
    {
        return m_binauralSamples;
    }

private:
    py::array_t<float, py::array::f_style> m_binauralSamples;
};

class InfiniteBinauralStreamer: public BinauralStreamer
{
public:
    InfiniteBinauralStreamer(CCore binauralRenderer)
    : BinauralStreamer(binauralRenderer)
    {
    }

    py::array_t<float, py::array::f_style> operator()(const SourceSamplesMap& samplesMap, const SourcePositionMap& positionMap, const std::optional<const Position>& listenerPosition = std::nullopt, const std::optional<const Orientation>& listenerOrientation = std::nullopt)
    {
        for (const auto& [source, samples] : samplesMap) {
            if (samples.size() > m_bufferSize) {
                throw std::invalid_argument("The length of the source samples cannot be larger than the buffer size.");
            }
        }
        py::array_t<float, py::array::f_style> binauralSamples({static_cast<py::ssize_t>(m_bufferSize), py::ssize_t(2)});
        binauralSamples[py::ellipsis()] = 0.f;
        auto binauralMem = binauralSamples.mutable_unchecked<2>();
        // Update listener position and orientation if given
        updateListenerPositionAndOrientation(m_binauralRenderer.GetListener(), listenerPosition, listenerOrientation);
        // Update sources
        for (const auto& source : m_binauralRenderer.GetSources()) {
            // Update source position if given
            updateSourcePosition(source, positionMap);
            // Process source samples if given
            if (samplesMap.find(source) != samplesMap.end()) {
                const auto& samples = samplesMap.find(source)->second;
                const py::ssize_t sourceSize = std::min(static_cast<py::ssize_t>(m_bufferSize), samples.size());
                processSourceSamples(source, samples, binauralMem.mutable_data(0, 0), binauralMem.mutable_data(0, 1), sourceSize, sourceSize);
            }
        }
        // Update environments
        processEnvironments(m_bufferSize, binauralMem.mutable_data(0, 0), binauralMem.mutable_data(0, 1));
        return binauralSamples;
    }
};


PYBIND11_MODULE(py3dti, m)
{
    m.doc() = "";

    py::class_<CListener, std::shared_ptr<CListener> >(m, "Listener")
        .def_property("position", [](const CListener& self) {
            const CVector3 v = self.GetListenerTransform().GetPosition();
            return std::make_tuple(v.x, v.y, v.z);
        }, [](CListener& self, const Position& position) {
            CTransform transform = self.GetListenerTransform();
            transform.SetPosition(CVector3(std::get<0>(position), std::get<1>(position), std::get<2>(position)));
            self.SetListenerTransform(transform);
        })
        .def_property("orientation", [](const CListener& self) {
            const CQuaternion q = self.GetListenerTransform().GetOrientation();
            return std::make_tuple(q.w, q.x, q.y, q.z);
        }, [](CListener& self, const Orientation& orientation) {
            CTransform transform = self.GetListenerTransform();
            transform.SetOrientation(CQuaternion(std::get<0>(orientation), std::get<1>(orientation), std::get<2>(orientation), std::get<3>(orientation)));
            self.SetListenerTransform(transform);
        })
        .def_property("head_radius", &CListener::GetHeadRadius, &CListener::SetHeadRadius)
        .def_property("ild_attenuation", &CListener::GetILDAttenuation, &CListener::SetILDAttenuation)
        .def("load_hrtf_from_sofa", [](const std::shared_ptr<CListener>& self, const std::string& sofaPath) {
            bool specifiedDelays;
            if (!HRTF::CreateFromSofa(sofaPath, self, specifiedDelays)) {
                throw std::runtime_error("Loading HRTF from SOFA file failed.");
            }
        }, "sofa_path"_a)
        .def("load_hrtf_from_sofa", [](const std::shared_ptr<CListener>& self, const std::filesystem::path& sofaPath) {
            bool specifiedDelays;
            if (!HRTF::CreateFromSofa(sofaPath.string(), self, specifiedDelays)) {
                throw std::runtime_error("Loading HRTF from SOFA file failed.");
            }
        }, "sofa_path"_a)
        .def("load_hrtf_from_3dti", [](const std::shared_ptr<CListener>& self, const std::string& threedtiPath) {
            if (!HRTF::CreateFrom3dti(threedtiPath, self)) {
                throw std::runtime_error("Loading HRTF from 3dti file failed.");
            }
        }, "3dti_path"_a)
        .def("load_hrtf_from_3dti", [](const std::shared_ptr<CListener>& self, const std::filesystem::path& threedtiPath) {
            if (!HRTF::CreateFrom3dti(threedtiPath.string(), self)) {
                throw std::runtime_error("Loading HRTF from 3dti file failed.");
            }
        }, "3dti_path"_a)
        .def("load_ild_near_field_effect_table", [](const std::shared_ptr<CListener>& self, const std::string& tablePath) {
            if (!ILD::CreateFrom3dti_ILDNearFieldEffectTable(tablePath, self)) {
                throw std::runtime_error("Loading ILD Near Field Effect configuration from 3dti file failed.");
            }
        }, "table_path"_a)
        .def("load_ild_near_field_effect_table", [](const std::shared_ptr<CListener>& self, const std::filesystem::path& tablePath) {
            if (!ILD::CreateFrom3dti_ILDNearFieldEffectTable(tablePath.string(), self)) {
                throw std::runtime_error("Loading ILD Near Field Effect configuration from 3dti file failed.");
            }
        }, "table_path"_a)
        .def("__repr__", [](const CListener& self) {
            std::ostringstream oss;
            oss << "<py3dti.Listener (" << &self << ") at position " << self.GetListenerTransform().GetPosition() << " with orientation " << self.GetListenerTransform().GetOrientation() << " >" << std::endl;
            return oss.str();
        })
    ;

    py::class_<CEnvironment, std::shared_ptr<CEnvironment> >(m, "Environment")
        .def("load_brir_from_sofa", [](const std::shared_ptr<CEnvironment>& self, const std::string& sofaPath) {
            if (!BRIR::CreateFromSofa(sofaPath, self)) {
                throw std::runtime_error("Loading BRIR from SOFA file failed.");
            };
        }, "sofa_path"_a)
        .def("load_brir_from_sofa", [](const std::shared_ptr<CEnvironment>& self, const std::filesystem::path& sofaPath) {
            if (!BRIR::CreateFromSofa(sofaPath.string(), self)) {
                throw std::runtime_error("Loading BRIR from SOFA file failed.");
            }
        }, "sofa_path"_a)
        .def("load_brir_from_3dti", [](const std::shared_ptr<CEnvironment>& self, const std::string& threedtiPath) {
            if (!BRIR::CreateFrom3dti(threedtiPath, self)) {
                throw std::runtime_error("Loading BRIR from 3dti file failed.");
            }
        }, "3dti_path"_a)
        .def("load_brir_from_3dti", [](const std::shared_ptr<CEnvironment>& self, const std::filesystem::path& threedtiPath) {
            if (!BRIR::CreateFrom3dti(threedtiPath.string(), self)) {
                throw std::runtime_error("Loading BRIR from 3dti file failed.");
            }
        }, "3dti_path"_a)
        .def("process_virtual_ambisonic_reverb", [](CEnvironment& self) {
            CMonoBuffer<float> leftBuffer;
            CMonoBuffer<float> rightBuffer;
            self.ProcessVirtualAmbisonicReverb(leftBuffer, rightBuffer);
            py::array_t<float> leftArray{static_cast<py::ssize_t>(leftBuffer.size()), &leftBuffer[0]};
            py::array_t<float> rightArray{static_cast<py::ssize_t>(rightBuffer.size()), &rightBuffer[0]};
            return std::make_pair(leftArray, rightArray);
        })
        .def("__repr__", [](const CEnvironment& self) {
            std::ostringstream oss;
            oss << "<py3dti.Environment (" << &self << ")>" << std::endl;
            return oss.str();
        })
    ;

    py::enum_<TSpatializationMode>(m, "SpatializationMode")
        .value("NO_SPATIALIZATION", TSpatializationMode::NoSpatialization)
        .value("HIGH_PERFORMANCE", TSpatializationMode::HighPerformance)
        .value("HIGH_QUALITY", TSpatializationMode::HighQuality)
        .export_values()
    ;

    py::class_<CSingleSourceDSP, std::shared_ptr<CSingleSourceDSP> >(m, "Source")
        .def_property("position", [](const CSingleSourceDSP& self) {
            const CVector3 v = self.GetCurrentSourceTransform().GetPosition();
            return std::make_tuple(v.x, v.y, v.z);
        }, [](CSingleSourceDSP& self, const Position& position) {
            CTransform transform = self.GetCurrentSourceTransform();
            transform.SetPosition(CVector3(std::get<0>(position), std::get<1>(position), std::get<2>(position)));
            self.SetSourceTransform(transform);
        })
        .def_property("spatialization_mode", &CSingleSourceDSP::GetSpatializationMode, &CSingleSourceDSP::SetSpatializationMode)
        .def_property("interpolation", &CSingleSourceDSP::IsInterpolationEnabled,
        [](CSingleSourceDSP& self, const bool value) {
            if (value) {
                self.EnableInterpolation();
            } else {
                self.DisableInterpolation();
            }
        })
        .def_property("anechoic_processing", &CSingleSourceDSP::IsAnechoicProcessEnabled,
        [](CSingleSourceDSP& self, const bool value) {
            if (value) {
                self.EnableAnechoicProcess();
            } else {
                self.DisableAnechoicProcess();
            }
        })
        .def_property("reverb_processing", &CSingleSourceDSP::IsReverbProcessEnabled,
        [](CSingleSourceDSP& self, const bool value) {
            if (value) {
                self.EnableReverbProcess();
            } else {
                self.DisableReverbProcess();
            }
        })
        .def_property("far_distance_effect", &CSingleSourceDSP::IsFarDistanceEffectEnabled,
        [](CSingleSourceDSP& self, const bool value) {
            if (value) {
                self.EnableFarDistanceEffect();
            } else {
                self.DisableFarDistanceEffect();
            }
        })
        .def_property("near_field_effect", &CSingleSourceDSP::IsNearFieldEffectEnabled,
        [](CSingleSourceDSP& self, const bool value) {
            if (value) {
                self.EnableNearFieldEffect();
            } else {
                self.DisableNearFieldEffect();
            }
        })
        .def_property("propagation_delay", &CSingleSourceDSP::IsPropagationDelayEnabled,
        [](CSingleSourceDSP& self, const bool value) {
            if (value) {
                self.EnablePropagationDelay();
            } else {
                self.DisablePropagationDelay();
            }
        })
        .def_property("anechoic_distance_attenuation", &CSingleSourceDSP::IsDistanceAttenuationEnabledAnechoic,
         [](CSingleSourceDSP& self, const bool value) {
             if (value) {
                 self.EnableDistanceAttenuationAnechoic();
             } else {
                 self.DisableDistanceAttenuationAnechoic();
             }
         })
        .def_property("anechoic_distance_attenuation_smoothing", &CSingleSourceDSP::IsDistanceAttenuationSmoothingEnabledAnechoic,
         [](CSingleSourceDSP& self, const bool value) {
             if (value) {
                 self.EnableDistanceAttenuationSmoothingAnechoic();
             } else {
                 self.DisableDistanceAttenuationSmoothingAnechoic();
             }
         })
        .def_property("reverb_distance_attenuation", &CSingleSourceDSP::IsDistanceAttenuationEnabledReverb,
         [](CSingleSourceDSP& self, const bool value) {
             if (value) {
                 self.EnableDistanceAttenuationReverb();
             } else {
                 self.DisableDistanceAttenuationReverb();
             }
         })
        .def("process_anechoic", [](CSingleSourceDSP& self, const py::array_t<float>& buffer) {
            const CMonoBuffer<float> inputBuffer{buffer.data(), buffer.data()+buffer.size()};
            self.SetBuffer(inputBuffer);
            CMonoBuffer<float> leftBuffer;
            CMonoBuffer<float> rightBuffer;
            self.ProcessAnechoic(leftBuffer, rightBuffer);
            py::array_t<float> leftArray{static_cast<py::ssize_t>(leftBuffer.size()), &leftBuffer[0]};
            py::array_t<float> rightArray{static_cast<py::ssize_t>(rightBuffer.size()), &rightBuffer[0]};
            return std::make_pair(leftArray, rightArray);
        })
        .def("__repr__", [](const CSingleSourceDSP& self) {
            std::ostringstream oss;
            oss << "<py3dti.Source (" << &self << ") at position " << self.GetCurrentSourceTransform().GetPosition() << ">" << std::endl;
            return oss.str();
        })
    ;

    py::class_<FiniteBinauralStreamer>(m, "FiniteBinauralStreamer")
        .def(py::init<CCore, SourceSamplesMap>(), "binaural_renderer"_a, "source_samples_map"_a)
        .def("__call__", &FiniteBinauralStreamer::operator(), "source_position_map"_a = SourcePositionMap(), "listener_position"_a = py::none(), "listener_orientation"_a = py::none())
        .def("__len__", &FiniteBinauralStreamer::size)
    ;

    py::class_<InfiniteBinauralStreamer>(m, "InfiniteBinauralStreamer")
        .def(py::init<CCore>(), "binaural_renderer"_a)
        .def("__call__", &InfiniteBinauralStreamer::operator(), "source_samples_map"_a, "source_position_map"_a = SourcePositionMap(), "listener_position"_a = py::none(), "listener_orientation"_a = py::none())
    ;

    py::class_<CCore>(m, "BinauralRenderer")
        .def(py::init([](const int sampleRate, const int bufferSize, const int resampledAngularResolution) {
            return CCore({sampleRate, bufferSize}, resampledAngularResolution);
        }), "rate"_a = 44100, "buffer_size"_a = 512, "resampled_angular_resolution"_a = 5)
        .def_property("rate", [](const CCore& self) {
            return self.GetAudioState().sampleRate;
        }, [](CCore& self, const int sampleRate) {
            TAudioStateStruct audioState = self.GetAudioState();
            audioState.sampleRate = sampleRate;
            self.SetAudioState(audioState);
        })
        .def_property("buffer_size", [](const CCore& self) {
            return self.GetAudioState().bufferSize;
        }, [](CCore& self, const int bufferSize) {
            TAudioStateStruct audioState = self.GetAudioState();
            audioState.bufferSize = bufferSize;
            self.SetAudioState(audioState);
        })
        .def_property("resampled_angular_resolution", &CCore::GetHRTFResamplingStep, &CCore::SetHRTFResamplingStep)
        .def("add_listener", [](CCore& self, const std::optional<const Position> position, const std::optional<const Orientation> orientation, const float headRadius) {
            if (self.GetListener() != nullptr) {
                throw std::runtime_error("BinauralRenderer already has a listener. Remove the previous one first.");
            }
            std::shared_ptr<CListener> listener = self.CreateListener(headRadius);
            updateListenerPositionAndOrientation(listener, position, orientation);
            return listener;
        }, "position"_a = py::none(), "orientation"_a = py::none(), "head_radius"_a = 0.0875)
        .def_property_readonly("listener", &CCore::GetListener)
        .def("add_source", [](CCore& self, const std::optional<const Position> position) {
            std::shared_ptr<CSingleSourceDSP> source = self.CreateSingleSourceDSP();
            updateSourcePosition(source, position);
            return source;
        }, "position"_a = py::none())
        .def_property_readonly("sources", &CCore::GetSources)
        .def("add_environment", &CCore::CreateEnvironment)
        .def_property_readonly("environments", &CCore::GetEnvironments)
        .def("render_offline", [](const CCore& self, const SourceSamplesMap& samplesMap, const SourcePositionsMap& positionsMap, const Positions& listenerPositions, const Orientations& listenerOrientations) {
            return OfflineFiniteBinauralStreamer(self, samplesMap, positionsMap, listenerPositions, listenerOrientations)();
        }, "source_samples_map"_a, "source_positions_map"_a = SourcePositionsMap(), "listener_positions"_a = Positions(), "listener_orientations"_a = Orientations())
        .def("render_online", [](const CCore& self) {
            return InfiniteBinauralStreamer(self);
        })
        .def("render_online", [](const CCore& self, const SourceSamplesMap& samplesMap) {
            return FiniteBinauralStreamer(self, samplesMap);
        }, "source_samples_map"_a)
        .def("__repr__", [](const CCore& self) {
            std::ostringstream oss;
            TAudioStateStruct audioState = self.GetAudioState();
            size_t numEnvironments = self.GetEnvironments().size();
            size_t numSources = self.GetSources().size();
            oss << "<py3dti.BinauralRenderer (" << &self << ") with buffer size "
            << audioState.bufferSize << ", sample rate " << audioState.sampleRate << "Hz, "
            << (self.GetListener() == nullptr ? "no" : "a") << " listener, "
            << numEnvironments << " environment" << (numEnvironments == 1 ? "" : "s")
            << " and " << numSources << " source" << (numSources == 1 ? "" : "s")
            << ">" << std::endl;
            return oss.str();
        })
    ;


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

