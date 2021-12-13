/*!
  * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
  * Licensed under the MIT License. See LICENSE file in the project root for license information.
  */
#ifndef LIGHTGBM_ENCODER_HPP_
#define LIGHTGBM_ENCODER_HPP_

#include <LightGBM/utils/json11.h>

#include <string>
#include <unordered_map>

namespace LightGBM {

using json11::Json;

  class CategoryFeatureEncoder {
  public:
    CategoryFeatureEncoder(const std::string feature_name) : feature_name_(feature_name){}

    std::string GetFeatureName() {
      return feature_name_;
    }

    virtual double Encode(double feature_value) const = 0;

    virtual json11::Json::object DumpToJsonObject() {
      json11::Json::object result {
        {encoder_type_key, json11::Json(default_encoder_type)},
        {feature_name_key, json11::Json(feature_name_)},
      };

      return result;
    }

  protected:
    std::string feature_name_;

    // property name keys
    const std::string feature_name_key = "feature_name";
    const std::string encoder_type_key = "encoder_type";

    // constant value
    const int default_encoder_type = 0;
  };

  class CategoryFeatureCountEncoder : public CategoryFeatureEncoder {
  public:
    CategoryFeatureCountEncoder(std::string feature_name, std::unordered_map<int, double> count_information) : CategoryFeatureEncoder(feature_name), count_information_(count_information){}

    double Encode(double feature_value);

    json11::Json::object DumpToJsonObject();

    // public constant value
    const int count_encoder_type = 1;

  private:
    std::unordered_map<int, double> count_information_;

    // property name keys
    const std::string count_information_key = "count_information";
    const std::string count_information_category_key = "cat";
    const std::string count_information_value_key = "value";

    // constant value
    const double default_value = 0.0;
  };

  class CategoryFeatureTargetEncoder : public CategoryFeatureEncoder {
  public:
    CategoryFeatureTargetEncoder(std::string feature_name, double prior, double prior_weight, double total_count, std::unordered_map<int, double> count_information)
      : CategoryFeatureEncoder(feature_name), prior_(prior), prior_weight_(prior_weight), total_count_(total_count), count_information_(count_information) {}

    double Encode(double feature_value);

    json11::Json::object DumpToJsonObject();

    // public constant value
    const int target_encoder_type = 1;

  private:
    std::unordered_map<int, double> count_information_;
    double prior_;
    double prior_weight_;
    double total_count_;

    // property name keys
    const std::string count_information_key = "count_information";
    const std::string count_information_category_key = "cat";
    const std::string count_information_value_key = "value";
    const std::string count_prior_key = "prior";
    const std::string count_prior_weight_key = "prior_weight";
    const std::string count_total_count_key = "total_count";

    // constant value
    const double default_value = 0.0;
  };

  class TargetCategoryInformationCollector {

  public:
    void HandleRecord(int fold_id, int feature_id, int category, double target);

  private:
    
  };

  class CategoryFeatureEncoderDeserializer {
  public:
    static CategoryFeatureEncoder* ParseFromJsonString(std::string content) {
      std::string error_message;
      json11::Json inputJson = json11::Json::parse(content, &error_message);

      return nullptr;
    }
  };
}

#endif
