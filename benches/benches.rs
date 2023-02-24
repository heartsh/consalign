extern crate consalign;
extern crate criterion;

use consalign::*;
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_consalign(criterion: &mut Criterion) {
  let input_file_path = Path::new(EXAMPLE_FASTA_FILE_PATH);
  let fasta_file_reader = Reader::from_file(Path::new(&input_file_path)).unwrap();
  let mut fasta_records = FastaRecords::new();
  let mut max_seq_len = 0;
  for x in fasta_file_reader.records() {
    let x = x.unwrap();
    let mut y = bytes2seq(x.seq());
    y.insert(0, PSEUDO_BASE);
    y.push(PSEUDO_BASE);
    let z = y.len();
    if z > max_seq_len {
      max_seq_len = z;
    }
    fasta_records.push(FastaRecord::new(String::from(x.id()), y));
  }
  let num_threads = num_cpus::get() as NumThreads;
  let mut thread_pool = Pool::new(num_threads);
  let output_dir_path = Path::new(OUTPUT_DIR_PATH);
  let disables_alifold = true;
  let disables_transplant = false;
  criterion.bench_function("wrapped_consalign::<u8, u8>", |x| {
    x.iter(|| {
      let _ = consalign_wrapped::<u8, u8>((
        &mut thread_pool,
        &fasta_records,
        output_dir_path,
        input_file_path,
        DEFAULT_BASEPAIR_PROB_TRAINED,
        DEFAULT_MATCH_PROB_TRAINED,
        ScoreModel::Ensemble,
        TrainType::TrainedTransfer,
        disables_alifold,
        DEFAULT_BASEPAIR_PROB_TURNER,
        DEFAULT_MATCH_PROB_TURNER,
        disables_transplant,
      ));
    });
  });
}

criterion_group!(benches, bench_consalign);
criterion_main!(benches);
