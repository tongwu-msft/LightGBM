#include "encoder.hpp"

namespace LightGBM {
  void CategoryFeatureTargetInformationCollector::HandleRecord(int fold_id, const std::vector<double>& record, double label) {
    auto& category_target_information = category_target_information_[fold_id];
    double a = record[1];
  }
}
