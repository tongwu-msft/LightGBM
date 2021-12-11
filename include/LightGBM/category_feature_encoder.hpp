/*!
  * Copyright (c) 2021 Microsoft Corporation. All rights reserved.
  * Licensed under the MIT License. See LICENSE file in the project root for license information.
  */
#ifndef LIGHTGBM_CATEGORY_CONVERTER_HPP_
#define LIGHTGBM_CATEGORY_CONVERTER_HPP_

#include <LightGBM/config.h>
#include <LightGBM/network.h>
#include <LightGBM/utils/json11.h>
#include <LightGBM/utils/text_reader.h>
#include <LightGBM/utils/threading.h>
#include <LightGBM/parser_base.h>

#include <algorithm>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace LightGBM {

using json11::Json;

class CategoryFeatureEncoder {
   protected:
    std::unordered_map<int, int> cat_fid_to_convert_fid_;

   public:
    virtual ~CategoryFeatureEncoder() {}

    virtual double CalcValue(const double sum_label, const double sum_count,
      const double all_fold_sum_count, const double prior) const = 0;

    virtual double CalcValue(const double sum_label, const double sum_count,
      const double all_fold_sum_count) const = 0;

    virtual std::string DumpToString() const = 0;

    virtual json11::Json DumpToJSONObject() const = 0;

    virtual std::string FeatureName() const = 0;

    virtual void SetPrior(const double /*prior*/, const double /*prior_weight*/) {}

    void SetCatFidToConvertFid(const std::unordered_map<int, int>& cat_fid_to_convert_fid) {
      cat_fid_to_convert_fid_ = cat_fid_to_convert_fid;
    }

    void RegisterConvertFid(const int cat_fid, const int convert_fid) {
      cat_fid_to_convert_fid_[cat_fid] = convert_fid;
    }

    int GetConvertFid(const int cat_fid) const {
      return cat_fid_to_convert_fid_.at(cat_fid);
    }

    static CategoryFeatureEncoder* CreateFromCharPointer(const char* char_pointer, size_t* used_len, double prior_weight);
  };


class TargetEncoder : public CategoryFeatureEncoder {
public:
  explicit TargetEncoder(const double prior) : prior_(prior) {}

  inline double CalcValue(const double sum_label, const double sum_count,
    const double /*all_fold_sum_count*/) const override {
    return (sum_label + prior_ * prior_weight_) / (sum_count + prior_weight_);
  }

  inline double CalcValue(const double sum_label, const double sum_count,
    const double /*all_fold_sum_count*/, const double /*prior*/) const override {
    return (sum_label + prior_ * prior_weight_) / (sum_count + prior_weight_);
  }

  void SetPrior(const double /*prior*/, const double prior_weight) override;

  std::string FeatureName() const override;

  json11::Json DumpToJSONObject() const override;

  std::string DumpToString() const override;

private:
  const double prior_;
  double prior_weight_;
};

class CountEncoder : public CategoryFeatureEncoder {
public:
  CountEncoder() {}

  inline double CalcValue(const double /*sum_label*/, const double /*sum_count*/,
    const double all_fold_sum_count) const override {
    return all_fold_sum_count;
  }

  inline double CalcValue(const double /*sum_label*/, const double /*sum_count*/,
    const double all_fold_sum_count, const double /*prior*/) const override {
    return all_fold_sum_count;
  }

  std::string FeatureName() const;

  json11::Json DumpToJSONObject() const;

  std::string DumpToString() const;
};

class TargetEncoderLabelMean : public CategoryFeatureEncoder {
public:
  TargetEncoderLabelMean() { prior_set_ = false; }

  void SetPrior(const double prior, const double prior_weight) override {
    prior_ = prior;
    prior_weight_ = prior_weight;
    prior_set_ = true;
  }

  inline double CalcValue(const double sum_label, const double sum_count,
    const double /*all_fold_sum_count*/) const override {
    if (!prior_set_) {
      Log::Fatal("TargetEncoderLabelMean is not ready since the prior value is not set.");
    }
    return (sum_label + prior_weight_ * prior_) / (sum_count + prior_weight_);
  }

  inline double CalcValue(const double sum_label, const double sum_count,
    const double /*all_fold_sum_count*/, const double prior) const override {
    if (!prior_set_) {
      Log::Fatal("TargetEncoderLabelMean is not ready since the prior value is not set.");
    }
    return (sum_label + prior * prior_weight_) / (sum_count + prior_weight_);
  }

  std::string FeatureName() const;

  json11::Json DumpToJSONObject() const;

  std::string DumpToString() const;

private:
  double prior_;
  double prior_weight_;
  bool prior_set_;
};

} // namespace LightGBM
#endif  // LIGHTGBM_CATEGORY_CONVERTER_HPP_
