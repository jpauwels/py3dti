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


PYBIND11_MODULE(py3dti, m)
{
    m.doc() = "";

    py::class_<CListener, std::shared_ptr<CListener> >(m, "Listener")
        .def_property("position", [](const CListener& self) {
            const CVector3 v = self.GetListenerTransform().GetPosition();
            return std::make_tuple(v.x, v.y, v.z);
        }, [](CListener& self, const std::tuple<float,float,float>& position) {
            CTransform transform = self.GetListenerTransform();
            transform.SetPosition(CVector3(std::get<0>(position), std::get<1>(position), std::get<2>(position)));
            self.SetListenerTransform(transform);
        })
        .def_property("orientation", [](const CListener& self) {
            const CQuaternion q = self.GetListenerTransform().GetOrientation();
            return std::make_tuple(q.w, q.x, q.y, q.z);
        }, [](CListener& self, const std::tuple<float,float,float,float>& orientation) {
            CTransform transform = self.GetListenerTransform();
            transform.SetOrientation(CQuaternion(std::get<0>(orientation), std::get<1>(orientation), std::get<2>(orientation), std::get<3>(orientation)));
            self.SetListenerTransform(transform);
        })
        .def_property("head_radius", &CListener::GetHeadRadius, &CListener::SetHeadRadius)
        .def_property("ild_attenuation", &CListener::GetILDAttenuation, &CListener::SetILDAttenuation)
        .def("load_hrtf_from_sofa", [](const std::shared_ptr<CListener>& self, const std::string& sofaPath) {
            bool specifiedDelays;
            if (!HRTF::CreateFromSofa(sofaPath, self, specifiedDelays)) {
                throw std::runtime_error("Loading HRTF from SOFA file failed");
            }
        }, "sofa_path"_a)
        .def("load_hrtf_from_sofa", [](const std::shared_ptr<CListener>& self, const std::filesystem::path& sofaPath) {
            bool specifiedDelays;
            if (!HRTF::CreateFromSofa(sofaPath.string(), self, specifiedDelays)) {
                throw std::runtime_error("Loading HRTF from SOFA file failed");
            }
        }, "sofa_path"_a)
        .def("load_hrtf_from_3dti", [](const std::shared_ptr<CListener>& self, const std::string& threedtiPath) {
            if (!HRTF::CreateFrom3dti(threedtiPath, self)) {
                throw std::runtime_error("Loading HRTF from 3dti file failed");
            }
        }, "3dti_path"_a)
        .def("load_hrtf_from_3dti", [](const std::shared_ptr<CListener>& self, const std::filesystem::path& threedtiPath) {
            if (!HRTF::CreateFrom3dti(threedtiPath.string(), self)) {
                throw std::runtime_error("Loading HRTF from 3dti file failed");
            }
        }, "3dti_path"_a)
        .def("load_ild_near_field_effect_table", [](const std::shared_ptr<CListener>& self, const std::string& tablePath) {
            if (!ILD::CreateFrom3dti_ILDNearFieldEffectTable(tablePath, self)) {
                throw std::runtime_error("Loading ILD Near Field Effect configuration from 3dti file failed");
            }
        }, "table_path"_a)
        .def("load_ild_near_field_effect_table", [](const std::shared_ptr<CListener>& self, const std::filesystem::path& tablePath) {
            if (!ILD::CreateFrom3dti_ILDNearFieldEffectTable(tablePath.string(), self)) {
                throw std::runtime_error("Loading ILD Near Field Effect configuration from 3dti file failed");
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
                throw std::runtime_error("Loading BRIR from SOFA file failed");
            };
        }, "sofa_path"_a)
        .def("load_brir_from_sofa", [](const std::shared_ptr<CEnvironment>& self, const std::filesystem::path& sofaPath) {
            if (!BRIR::CreateFromSofa(sofaPath.string(), self)) {
                throw std::runtime_error("Loading BRIR from SOFA file failed");
            }
        }, "sofa_path"_a)
        .def("load_brir_from_3dti", [](const std::shared_ptr<CEnvironment>& self, const std::string& threedtiPath) {
            if (!BRIR::CreateFrom3dti(threedtiPath, self)) {
                throw std::runtime_error("Loading BRIR from 3dti file failed");
            }
        }, "3dti_path"_a)
        .def("load_brir_from_3dti", [](const std::shared_ptr<CEnvironment>& self, const std::filesystem::path& threedtiPath) {
            if (!BRIR::CreateFrom3dti(threedtiPath.string(), self)) {
                throw std::runtime_error("Loading BRIR from 3dti file failed");
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
        }, [](CSingleSourceDSP& self, const std::tuple<float,float,float>& position) {
            CTransform transform = self.GetCurrentSourceTransform();
            transform.SetPosition(CVector3(std::get<0>(position), std::get<1>(position), std::get<2>(position)));
            self.SetSourceTransform(transform);
        })
        .def_property("orientation", [](const CSingleSourceDSP& self) {
            const CQuaternion q = self.GetCurrentSourceTransform().GetOrientation();
            return std::make_tuple(q.w, q.x, q.y, q.z);
        }, [](CSingleSourceDSP& self, const std::tuple<float,float,float,float>& orientation) {
            CTransform transform = self.GetCurrentSourceTransform();
            transform.SetOrientation(CQuaternion(std::get<0>(orientation), std::get<1>(orientation), std::get<2>(orientation), std::get<3>(orientation)));
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
            oss << "<py3dti.Source (" << &self << ") at position " << self.GetCurrentSourceTransform().GetPosition() << " with orientation " << self.GetCurrentSourceTransform().GetOrientation() << " >" << std::endl;
            return oss.str();
        })
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
        .def("add_listener", [](CCore& self, const std::optional<const std::tuple<float,float,float>> position, const std::optional<const std::tuple<float,float,float,float>> orientation, const float headRadius) {
            if (self.GetListener() != nullptr) {
                throw std::runtime_error("BinauralRenderer already has a listener. Remove the previous one first.");
            }
            std::shared_ptr<CListener> listener = self.CreateListener(headRadius);
            if (position || orientation) {
                CTransform transform = listener->GetListenerTransform();
                if (position) {
                    transform.SetPosition(CVector3(std::get<0>(*position), std::get<1>(*position), std::get<2>(*position)));
                }
                if (orientation) {
                    transform.SetOrientation(CQuaternion(std::get<0>(*orientation), std::get<1>(*orientation), std::get<2>(*orientation), std::get<3>(*orientation)));
                }
                listener->SetListenerTransform(transform);
            }
            return listener;
        }, "position"_a = py::none(), "orientation"_a = py::none(), "head_radius"_a =  0.0875)
        .def_property_readonly("listener", &CCore::GetListener)
        .def("add_source", [](CCore& self, const std::optional<const std::tuple<float,float,float>> position, const std::optional<const std::tuple<float,float,float,float>> orientation) {
            std::shared_ptr<CSingleSourceDSP> source = self.CreateSingleSourceDSP();
            if (position || orientation) {
                CTransform transform = source->GetCurrentSourceTransform();
                if (position) {
                    transform.SetPosition(CVector3(std::get<0>(*position), std::get<1>(*position), std::get<2>(*position)));
                }
                if (orientation) {
                    transform.SetOrientation(CQuaternion(std::get<0>(*orientation), std::get<1>(*orientation), std::get<2>(*orientation), std::get<3>(*orientation)));
                }
                source->SetSourceTransform(transform);
            }
            return source;
        }, "position"_a = py::none(), "orientation"_a = py::none())
        .def_property_readonly("sources", &CCore::GetSources)
        .def("add_environment", &CCore::CreateEnvironment)
        .def_property_readonly("environments", &CCore::GetEnvironments)
        .def("render_offline", [](const CCore& self, const std::map<const std::shared_ptr<CSingleSourceDSP>, const py::array_t<float>>& sampleMap) {
            std::vector<py::ssize_t> sourceLengths;
            for (const auto& kv : sampleMap) {
                sourceLengths.push_back(kv.second.size());
            }
            const py::ssize_t binauralLength = *std::max_element(sourceLengths.begin(), sourceLengths.end());
            py::array_t<float, py::array::f_style> binauralSamples({binauralLength, py::ssize_t(2)});
            binauralSamples[py::ellipsis()] = 0.f;
            const int bufferSize = self.GetAudioState().bufferSize;
            CMonoBuffer<float> inputBuffer(bufferSize);
            CMonoBuffer<float> leftBuffer(bufferSize);
            CMonoBuffer<float> rightBuffer(bufferSize);
            auto binauralMem = binauralSamples.mutable_unchecked<2>();
            py::ssize_t start = 0;
            // Helper function for adding output buffers to contents of stereo file
            auto addToOutput = [&start, &leftBuffer, &rightBuffer, &binauralMem](const py::ssize_t size) {
                std::transform(leftBuffer.begin(), leftBuffer.begin()+size, binauralMem.mutable_data(start, 0), binauralMem.mutable_data(start, 0), std::plus<float>());
                std::transform(rightBuffer.begin(), rightBuffer.begin()+size, binauralMem.mutable_data(start, 1), binauralMem.mutable_data(start, 1), std::plus<float>());
            };
            for (; start < binauralLength; start += bufferSize) {
                const py::ssize_t blockEnd = std::min(start + bufferSize, binauralLength);
                for (const auto& [source, samples] : sampleMap) {
                    if (start < samples.size()) {
                        const py::ssize_t sourceEnd = std::min(blockEnd, samples.size());
                        const py::ssize_t sourceSize = sourceEnd - start;
                        const auto samplesMemory = samples.unchecked<1>();
                        std::copy(samplesMemory.data(start), samplesMemory.data(sourceEnd), inputBuffer.begin());
                        std::fill(inputBuffer.begin()+sourceSize, inputBuffer.end(), 0.f);
                        source->SetBuffer(inputBuffer);
                        source->ProcessAnechoic(leftBuffer, rightBuffer);
                        addToOutput(sourceSize);
                    }
                }
                const py::ssize_t blockSize = blockEnd - start;
                for (const auto& environment : self.GetEnvironments()) {
                    environment->ProcessVirtualAmbisonicReverb(leftBuffer, rightBuffer);
                    addToOutput(blockSize);
                }
            }
            return binauralSamples;
        }, "sample_map"_a)
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

