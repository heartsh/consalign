use FeatureCountSetsPosterior;
impl FeatureCountSetsPosterior {
pub fn load_trained_score_params() -> FeatureCountSetsPosterior {
FeatureCountSetsPosterior {
  basepair_count_posterior: 8.,
  align_count_posterior: 8.,
  basepair_count_offset: -2.3,
  align_count_offset: -2.3,
}
}
}
