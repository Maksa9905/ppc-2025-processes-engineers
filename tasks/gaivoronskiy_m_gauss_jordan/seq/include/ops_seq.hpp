#pragma once

#include "gaivoronskiy_m_gauss_jordan/common/include/common.hpp"
#include "task/include/task.hpp"

namespace gaivoronskiy_m_gauss_jordan {

class GaivoronskiyMGaussJordanSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit GaivoronskiyMGaussJordanSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace gaivoronskiy_m_gauss_jordan
