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
using MonoInput = py::array_t<float, py::array::c_style | py::array::forcecast>;
using BinauralOutput = py::array_t<float, py::array::f_style>;
using Size = py::ssize_t;
typedef std::map<const std::shared_ptr<CSingleSourceDSP>, const MonoInput> SourceSamplesMap;
typedef std::map<const std::shared_ptr<CSingleSourceDSP>, const Position> SourcePositionMap;
typedef std::map<const std::shared_ptr<CSingleSourceDSP>, const Positions> SourcePositionsMap;
typedef std::map<const std::shared_ptr<CSingleSourceDSP>, const double> SourceOffsetDurationMap;
typedef std::map<const std::shared_ptr<CSingleSourceDSP>, const Size> SourceOffsetSamplesMap;


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
    BinauralStreamer(const std::shared_ptr<CCore> binauralRenderer)
    : m_binauralRenderer(binauralRenderer)
    , m_bufferSize(binauralRenderer->GetAudioState().bufferSize)
    , m_inputBuffer(m_bufferSize)
    , m_leftBuffer(m_bufferSize)
    , m_rightBuffer(m_bufferSize)
    , m_start(0)
    {
    }

protected:
    void processSourceSamples(const std::shared_ptr<CSingleSourceDSP>& source, const MonoInput& samples, float* const leftPtr, float* const rightPtr, const Size sourceStart, const Size sourceSize)
    {
        const auto samplesMemory = samples.unchecked<1>();
        std::copy(samplesMemory.data(sourceStart), samplesMemory.data(sourceStart+sourceSize), m_inputBuffer.begin());
        std::fill(m_inputBuffer.begin()+sourceSize, m_inputBuffer.end(), 0.f);
        source->SetBuffer(m_inputBuffer);
        source->ProcessAnechoic(m_leftBuffer, m_rightBuffer);
        addToOutput(sourceSize, leftPtr, rightPtr);
    }

    void processEnvironments(const Size size, float* const leftPtr, float* const rightPtr) {
        for (const auto& environment : m_binauralRenderer->GetEnvironments()) {
            environment->ProcessVirtualAmbisonicReverb(m_leftBuffer, m_rightBuffer);
            addToOutput(size, leftPtr, rightPtr);
        }
    }

    void addToOutput(const Size size, float* const leftPtr, float* const rightPtr)
    {
        std::transform(m_leftBuffer.begin(), m_leftBuffer.begin()+size, leftPtr, leftPtr, std::plus<float>());
        std::transform(m_rightBuffer.begin(), m_rightBuffer.begin()+size, rightPtr, rightPtr, std::plus<float>());
    }

    const std::shared_ptr<CCore> m_binauralRenderer;
    const int m_bufferSize;
    CMonoBuffer<float> m_inputBuffer;
    CMonoBuffer<float> m_leftBuffer;
    CMonoBuffer<float> m_rightBuffer;
    Size m_start;
};

class FiniteBinauralStreamer: public BinauralStreamer
{
public:
    FiniteBinauralStreamer(const std::shared_ptr<CCore>& binauralRenderer, const SourceSamplesMap& samplesMap, const SourceOffsetDurationMap& offsetMap = SourceOffsetDurationMap())
    : BinauralStreamer(binauralRenderer)
    , m_samplesMap(samplesMap)
    {
        if (samplesMap.empty()) {
            throw std::invalid_argument("At least one source with associated audio samples is required.");
        }
        std::vector<Size> sourceLengths;
        const int sampleRate = binauralRenderer->GetAudioState().sampleRate;
        for (const auto& [source, samples] : samplesMap) {
            Size offsetSamples = 0;
            const auto& offsetItem = offsetMap.find(source);
            if (offsetItem != offsetMap.end()) {
                offsetSamples = std::round(offsetItem->second * sampleRate);
            }
            m_offsetMap.emplace(source, offsetSamples);
            sourceLengths.push_back(samples.size() + offsetSamples);
        }
        m_binauralLength = *std::max_element(sourceLengths.begin(), sourceLengths.end());
    }

    size_t size() const
    {
        return std::ceil(static_cast<double>(m_binauralLength) / m_bufferSize);
    }

    BinauralOutput operator()(const SourcePositionMap& positionMap, const std::optional<const Position>& listenerPosition = std::nullopt, const std::optional<const Orientation>& listenerOrientation = std::nullopt)
    {
        if (m_start >= m_binauralLength) {
            throw py::stop_iteration("All source samples have been processed.");
        }
        BinauralOutput binauralSamples({static_cast<Size>(m_bufferSize), Size(2)});
        binauralSamples[py::ellipsis()] = 0.f;
        auto binauralMem = binauralSamples.mutable_unchecked<2>();
        // Update listener position and orientation if given
        updateListenerPositionAndOrientation(m_binauralRenderer->GetListener(), listenerPosition, listenerOrientation);
        // Update sources
        const Size nextStart = m_start + m_bufferSize;
        for (const auto& [source, samples] : m_samplesMap) {
            // Update source position if given
            updateSourcePosition(source, positionMap);
            // Process source samples if any still left
            processSourceSamples(source, samples, binauralMem.mutable_data(0, 0), binauralMem.mutable_data(0, 1), nextStart);
        }
        // Update environments
        processEnvironments(m_bufferSize, binauralMem.mutable_data(0, 0), binauralMem.mutable_data(0, 1));
        m_start += m_bufferSize;
        return binauralSamples;
    }

protected:
    using BinauralStreamer::processSourceSamples;

    void processSourceSamples(const std::shared_ptr<CSingleSourceDSP>& source, const MonoInput& samples, float* const leftPtr, float* const rightPtr, const Size nextStart) {
        const Size offset = m_offsetMap[source];
        if (nextStart > offset && m_start < samples.size() + offset) {
            const Size sourceEnd = std::min(nextStart - offset, samples.size());
            const Size sourceStart = std::max(m_start - offset, static_cast<Size>(0));
            const Size sourceSize = sourceEnd - sourceStart;
            processSourceSamples(source, samples, leftPtr, rightPtr, sourceStart, sourceSize);
        }
    }

    Size m_binauralLength;

private:
    const SourceSamplesMap m_samplesMap;
    SourceOffsetSamplesMap m_offsetMap;
};

class OfflineFiniteBinauralStreamer: public FiniteBinauralStreamer
{
public:
    OfflineFiniteBinauralStreamer(const std::shared_ptr<CCore>& binauralRenderer, const SourceSamplesMap& samplesMap, const SourcePositionsMap& positionsMap = SourcePositionsMap(), const Positions& listenerPositions = Positions(), const Orientations& listenerOrientations = Orientations(), const SourceOffsetDurationMap& offsetMap = SourceOffsetDurationMap())
    : FiniteBinauralStreamer(binauralRenderer, samplesMap, offsetMap)
    , m_binauralSamples({m_binauralLength, Size(2)})
    {
        m_binauralSamples[py::ellipsis()] = 0.f;
        auto binauralMem = m_binauralSamples.mutable_unchecked<2>();
        for (size_t blockIdx = 0; m_start < m_binauralLength; m_start += m_bufferSize, ++blockIdx) {
            // Update listener position and orientation if given
            updateListenerPositionAndOrientation(m_binauralRenderer->GetListener(), blockIdx, listenerPositions, listenerOrientations);
            // Update sources
            const Size nextStart = std::min(m_start + m_bufferSize, m_binauralLength);
            for (const auto& [source, samples] : samplesMap) {
                // Update source position if given
                updateSourcePosition(source, blockIdx, positionsMap);
                // Process source samples if any still left
                processSourceSamples(source, samples, binauralMem.mutable_data(m_start, 0), binauralMem.mutable_data(m_start, 1), nextStart);
            }
            // Update environments
            const Size blockSize = nextStart - m_start;
            processEnvironments(blockSize, binauralMem.mutable_data(m_start, 0), binauralMem.mutable_data(m_start, 1));
        }
    }

    const BinauralOutput& operator()()
    {
        return m_binauralSamples;
    }

private:
    BinauralOutput m_binauralSamples;
};

class InfiniteBinauralStreamer: public BinauralStreamer
{
public:
    InfiniteBinauralStreamer(const std::shared_ptr<CCore>& binauralRenderer)
    : BinauralStreamer(binauralRenderer)
    {
    }

    BinauralOutput operator()(const SourceSamplesMap& samplesMap, const SourcePositionMap& positionMap, const std::optional<const Position>& listenerPosition = std::nullopt, const std::optional<const Orientation>& listenerOrientation = std::nullopt)
    {
        for (const auto& [source, samples] : samplesMap) {
            if (samples.size() > m_bufferSize) {
                throw std::invalid_argument("The length of the source samples cannot be larger than the buffer size.");
            }
        }
        BinauralOutput binauralSamples({static_cast<Size>(m_bufferSize), Size(2)});
        binauralSamples[py::ellipsis()] = 0.f;
        auto binauralMem = binauralSamples.mutable_unchecked<2>();
        // Update listener position and orientation if given
        updateListenerPositionAndOrientation(m_binauralRenderer->GetListener(), listenerPosition, listenerOrientation);
        // Update sources
        for (const auto& source : m_binauralRenderer->GetSources()) {
            // Update source position if given
            updateSourcePosition(source, positionMap);
            // Process source samples if given
            if (samplesMap.find(source) != samplesMap.end()) {
                const auto& samples = samplesMap.find(source)->second;
                const Size sourceSize = std::min(static_cast<Size>(m_bufferSize), samples.size());
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
        .def_property("head_radius", [](const CListener& self) -> std::optional<float> {
            if (self.IsCustomizedITDEnabled()) {
                return self.GetHeadRadius();
            } else {
                return std::nullopt;
            }
        }, [](CListener& self, const std::optional<float> headRadius) {
            if (headRadius) {
                self.SetHeadRadius(*headRadius);
                self.EnableCustomizedITD();
            } else {
                self.DisableCustomizedITD();
            }
        })
        .def_property("ild_attenuation", &CListener::GetILDAttenuation, &CListener::SetILDAttenuation)
        .def("load_hrtf_from_sofa", [](const std::shared_ptr<CListener>& self, const std::filesystem::path& sofaPath) {
            bool specifiedDelays;
            if (!HRTF::CreateFromSofa(sofaPath.string(), self, specifiedDelays)) {
                throw std::runtime_error("Loading HRTF from SOFA file failed.");
            }
        }, "sofa_path"_a)
        .def("load_hrtf_from_3dti", [](const std::shared_ptr<CListener>& self, const std::filesystem::path& threedtiPath) {
            if (!HRTF::CreateFrom3dti(threedtiPath.string(), self)) {
                throw std::runtime_error("Loading HRTF from 3dti file failed.");
            }
        }, "3dti_path"_a)
        .def("load_ild_near_field_effect_table", [](const std::shared_ptr<CListener>& self, const std::filesystem::path& tablePath) {
            if (!ILD::CreateFrom3dti_ILDNearFieldEffectTable(tablePath.string(), self)) {
                throw std::runtime_error("Loading ILD Near Field Effect configuration from 3dti file failed.");
            }
        }, "table_path"_a)
        .def("__repr__", [](const CListener& self) {
            std::ostringstream oss;
            oss << "<py3dti.Listener (" << &self << ") at position " << self.GetListenerTransform().GetPosition() << " with orientation " << self.GetListenerTransform().GetOrientation();
            if (self.IsCustomizedITDEnabled()) {
                oss.precision(4);
                oss << " and a head radius of " << self.GetHeadRadius() << " m";
            }
            oss << ">";
            return oss.str();
        })
    ;

    py::class_<CEnvironment, std::shared_ptr<CEnvironment> >(m, "Environment")
        .def("load_brir_from_sofa", [](const std::shared_ptr<CEnvironment>& self, const std::filesystem::path& sofaPath) {
            if (!BRIR::CreateFromSofa(sofaPath.string(), self)) {
                throw std::runtime_error("Loading BRIR from SOFA file failed.");
            }
        }, "sofa_path"_a)
        .def("load_brir_from_3dti", [](const std::shared_ptr<CEnvironment>& self, const std::filesystem::path& threedtiPath) {
            if (!BRIR::CreateFrom3dti(threedtiPath.string(), self)) {
                throw std::runtime_error("Loading BRIR from 3dti file failed.");
            }
        }, "3dti_path"_a)
        .def("process_virtual_ambisonic_reverb", [](CEnvironment& self) {
            CMonoBuffer<float> leftBuffer;
            CMonoBuffer<float> rightBuffer;
            self.ProcessVirtualAmbisonicReverb(leftBuffer, rightBuffer);
            py::array_t<float> leftArray{static_cast<Size>(leftBuffer.size()), leftBuffer.data()};
            py::array_t<float> rightArray{static_cast<Size>(rightBuffer.size()), rightBuffer.data()};
            return std::make_pair(leftArray, rightArray);
        })
        .def("__repr__", [](const CEnvironment& self) {
            std::ostringstream oss;
            oss << "<py3dti.Environment (" << &self << ")>";
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
        .def("process_anechoic", [](CSingleSourceDSP& self, const MonoInput& buffer) {
            const CMonoBuffer<float> inputBuffer{buffer.data(), buffer.data(buffer.size())};
            self.SetBuffer(inputBuffer);
            CMonoBuffer<float> leftBuffer;
            CMonoBuffer<float> rightBuffer;
            self.ProcessAnechoic(leftBuffer, rightBuffer);
            py::array_t<float> leftArray{static_cast<Size>(leftBuffer.size()), leftBuffer.data()};
            py::array_t<float> rightArray{static_cast<Size>(rightBuffer.size()), rightBuffer.data()};
            return std::make_pair(leftArray, rightArray);
        })
        .def("__repr__", [](const CSingleSourceDSP& self) {
            std::ostringstream oss;
            oss << "<py3dti.Source (" << &self << ") at position " << self.GetCurrentSourceTransform().GetPosition() << ">";
            return oss.str();
        })
    ;

    py::class_<FiniteBinauralStreamer>(m, "FiniteBinauralStreamer")
        .def(py::init<const std::shared_ptr<CCore>&, SourceSamplesMap, SourceOffsetDurationMap>(), "binaural_renderer"_a, "source_samples_map"_a, "source_offset_map"_a = SourceOffsetDurationMap())
        .def("__call__", &FiniteBinauralStreamer::operator(), "source_position_map"_a = SourcePositionMap(), "listener_position"_a = py::none(), "listener_orientation"_a = py::none())
        .def("__len__", &FiniteBinauralStreamer::size)
    ;

    py::class_<InfiniteBinauralStreamer>(m, "InfiniteBinauralStreamer")
        .def(py::init<const std::shared_ptr<CCore>&>(), "binaural_renderer"_a)
        .def("__call__", &InfiniteBinauralStreamer::operator(), "source_samples_map"_a, "source_position_map"_a = SourcePositionMap(), "listener_position"_a = py::none(), "listener_orientation"_a = py::none())
    ;

    py::class_<CCore, std::shared_ptr<CCore>>(m, "BinauralRenderer")
        .def(py::init([](const int sampleRate, const int bufferSize, const int resampledAngularResolution, const std::optional<const Position> position, const std::optional<const Orientation> orientation, const float headRadius) {
            TAudioStateStruct state{sampleRate, bufferSize};
            auto core = std::make_shared<CCore>(state, resampledAngularResolution);
            auto listener = core->CreateListener(headRadius);
            updateListenerPositionAndOrientation(listener, position, orientation);
            return core;
        }), "rate"_a = 44100, "buffer_size"_a = 512, "resampled_angular_resolution"_a = 5, "position"_a = py::none(), "orientation"_a = py::none(), "head_radius"_a =  0.0875)
        .def_property("rate", [](const CCore& self) {
            return self.GetAudioState().sampleRate;
        }, [](CCore& self, const int sampleRate) {
            TAudioStateStruct audioState = self.GetAudioState();
            audioState.sampleRate = sampleRate;
            self.SetAudioState(audioState);
        })
        .def_property_readonly("buffer_size", [](const CCore& self) {
            return self.GetAudioState().bufferSize;
        })
        .def_property("resampled_angular_resolution", &CCore::GetHRTFResamplingStep, &CCore::SetHRTFResamplingStep)
        .def_property_readonly("listener", py::cpp_function(&CCore::GetListener, py::keep_alive<0, 1>()))
        .def("add_source", [](CCore& self, const std::optional<const Position> position) {
            std::shared_ptr<CSingleSourceDSP> source = self.CreateSingleSourceDSP();
            updateSourcePosition(source, position);
            return source;
        }, "position"_a = py::none(), py::keep_alive<0, 1>())
        .def_property_readonly("sources", &CCore::GetSources)
        .def("add_environment", &CCore::CreateEnvironment, py::keep_alive<0, 1>())
        .def_property_readonly("environments", &CCore::GetEnvironments)
        .def("render_offline", [](const std::shared_ptr<CCore>& self, const SourceSamplesMap& samplesMap, const SourcePositionsMap& positionsMap, const Positions& listenerPositions, const Orientations& listenerOrientations, const SourceOffsetDurationMap& offsetMap) {
            return OfflineFiniteBinauralStreamer(self, samplesMap, positionsMap, listenerPositions, listenerOrientations, offsetMap)();
        }, "source_samples_map"_a, "source_positions_map"_a = SourcePositionsMap(), "listener_positions"_a = Positions(), "listener_orientations"_a = Orientations(), "source_offset_map"_a = SourceOffsetDurationMap())
        .def("render_online", [](const std::shared_ptr<CCore>& self) {
            return InfiniteBinauralStreamer(self);
        })
        .def("render_online", [](const std::shared_ptr<CCore>& self, const SourceSamplesMap& samplesMap, const SourceOffsetDurationMap& offsetMap) {
            return FiniteBinauralStreamer(self, samplesMap, offsetMap);
        }, "source_samples_map"_a, "source_offset_map"_a = SourceOffsetDurationMap())
        .def("__repr__", [](const CCore& self) {
            std::ostringstream oss;
            TAudioStateStruct audioState = self.GetAudioState();
            size_t numEnvironments = self.GetEnvironments().size();
            size_t numSources = self.GetSources().size();
            oss << "<py3dti.BinauralRenderer (" << &self << ") with buffer size "
            << audioState.bufferSize << ", sample rate " << audioState.sampleRate << "Hz, "
            << numEnvironments << " environment" << (numEnvironments == 1 ? "" : "s")
            << " and " << numSources << " source" << (numSources == 1 ? "" : "s")
            << ">";
            return oss.str();
        })
    ;

    m.def("taitbryan2quaternion", [](const float yaw, const float pitch, const float roll) {
        const CQuaternion q = CQuaternion::FromYawPitchRoll(yaw, pitch, roll);
        return std::make_tuple(q.w, q.x, q.y, q.z);
    }, "yaw"_a, "pitch"_a, "roll"_a);
    m.def("quaternion2taitbryan", [](const float scalar, const float backFront, const float rightLeft, const float downUp) {
        const CQuaternion q(scalar, backFront, rightLeft, downUp);
        float yaw, pitch, roll;
        q.ToYawPitchRoll(yaw, pitch, roll);
        return std::make_tuple(yaw, pitch, roll);
    }, "scalar"_a, "back_front"_a, "right_left"_a, "down_up"_a);
    m.def("axisangle2quaternion", [](const float backFront, const float rightLeft, const float downUp, const float angle) {
        const CQuaternion q = CQuaternion::FromAxisAngle(CVector3(backFront, rightLeft, downUp), angle);
        return std::make_tuple(q.w, q.x, q.y, q.z);
    }, "back_front"_a, "right_left"_a, "down_up"_a, "angle"_a);
    m.def("quaternion2axisangle", [](const float scalar, const float backFront, const float rightLeft, const float downUp) {
        const CQuaternion q(scalar, backFront, rightLeft, downUp);
        CVector3 axis;
        float angle;
        q.ToAxisAngle(axis, angle);
        return std::make_tuple(axis.x, axis.y, axis.z, angle);
    }, "scalar"_a, "back_front"_a, "right_left"_a, "down_up"_a);


#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}

