/*!
  * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
  * Licensed under the MIT License. See LICENSE file in the project root for license information.
  */

#include <LightGBM/utils/threading.h>
#include <LightGBM/category_feature_encoder.hpp>

namespace LightGBM {
  std::string TargetEncoderLabelMean::FeatureName() const {
    std::stringstream str_stream;
    Common::C_stringstream(str_stream);
    str_stream << "label_mean_prior_target_encoding_" << prior_;
    return str_stream.str();
  }

  json11::Json TargetEncoderLabelMean::DumpToJSONObject() const {
    json11::Json::array cat_fid_to_convert_fid_array;
    for (const auto& pair : cat_fid_to_convert_fid_) {
      cat_fid_to_convert_fid_array.emplace_back(
        json11::Json::object{
          {"cat_fid", json11::Json(pair.first)},
          {"convert_fid", json11::Json(pair.second)
        }
        });
    }

    json11::Json ret(json11::Json::object{
      {"name", json11::Json("target_encoder")},
      {"prior", json11::Json(prior_)},
      {"categorical_feature_index_to_encoded_feature_index", json11::Json(cat_fid_to_convert_fid_array)}
      });
    return ret;
  }

  std::string TargetEncoderLabelMean::DumpToString() const {
    std::stringstream str_stream;
    Common::C_stringstream(str_stream);
    str_stream << "type=target_encoder_label_mean\n";
    str_stream << "prior=" << prior_ << "\n";
    str_stream << "categorical_feature_index_to_encoded_feature_index=" <<
#if ((defined(sun) || defined(__sun)) && (defined(__SVR4) || defined(__svr4__)))
      CommonLegacy::UnorderedMapToString<false, false, int, int>(cat_fid_to_convert_fid_, ':', ' ') << "\n";
#else
      CommonC::UnorderedMapToString<false, false, int, int>(cat_fid_to_convert_fid_, ':', ' ') << "\n";
#endif  // ((defined(sun) || defined(__sun)) && (defined(__SVR4) || defined(__svr4__)))
    return str_stream.str();
  }
} 
