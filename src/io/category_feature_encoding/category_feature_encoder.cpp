/*!
  * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
  * Licensed under the MIT License. See LICENSE file in the project root for license information.
  */

#include <LightGBM/utils/threading.h>
#include <LightGBM/category_feature_encoder.hpp>

namespace LightGBM {

  CategoryFeatureEncoder* CategoryFeatureEncoder::CreateFromCharPointer(const char* char_pointer, size_t* used_len, double prior_weight) {
    const char* char_pointer_start = char_pointer;
    size_t line_len = Common::GetLine(char_pointer);
    std::string line(char_pointer, line_len);
    char_pointer += line_len;
    char_pointer = Common::SkipNewLine(char_pointer);
    CategoryFeatureEncoder* ret = nullptr;
    if (Common::StartsWith(line, "type=")) {
      std::string type = Common::Split(line.c_str(), "=")[1];

      if (type == std::string("target_encoder") || type == std::string("target_encoder_label_mean")) {
        line_len = Common::GetLine(char_pointer);
        line = std::string(char_pointer, line_len);
        char_pointer += line_len;
        char_pointer = Common::SkipNewLine(char_pointer);
        if (!Common::StartsWith(line.c_str(), "prior=")) {
          Log::Fatal("CategoryFeatureEncoder model format error");
        }
        double prior = 0.0f;
        Common::Atof(Common::Split(line.c_str(), "=")[1].c_str(), &prior);
        if (type == std::string("target_encoder")) {
          ret = new TargetEncoder(prior);
        } else {
          ret = new TargetEncoderLabelMean();
        }
        ret->SetPrior(prior, prior_weight);
      } else if (type == std::string("count_encoder")) {
        ret = new CountEncoder();
      } else {
        Log::Fatal("Unknown CategoryFeatureEncoder type %s", type.c_str());
      }

      line_len = Common::GetLine(char_pointer);
      line = std::string(char_pointer, line_len);
      char_pointer += line_len;
      char_pointer = Common::SkipNewLine(char_pointer);
      if (!Common::StartsWith(line.c_str(), "categorical_feature_index_to_encoded_feature_index=")) {
        Log::Fatal("CategoryFeatureEncoder model format error");
      }
      std::vector<std::string> feature_index_pair = Common::Split(Common::Split(line.c_str(), "=")[1].c_str(), " ");
      ret->cat_fid_to_convert_fid_.clear();
      for (auto& pair_string : feature_index_pair) {
        std::vector<std::string> cat_fid_and_convert_fid_string = Common::Split(pair_string.c_str(), ":");
        int cat_fid = 0;
        Common::Atoi(cat_fid_and_convert_fid_string[0].c_str(), &cat_fid);
        int convert_fid = 0;
        Common::Atoi(cat_fid_and_convert_fid_string[1].c_str(), &convert_fid);
        ret->cat_fid_to_convert_fid_[cat_fid] = convert_fid;
      }
      *used_len = static_cast<size_t>(char_pointer - char_pointer_start);
    } else {
      Log::Fatal("CategoryFeatureEncoder model format error");
    }
    return ret;
  }
} 
