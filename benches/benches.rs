extern crate consalign;
extern crate criterion;

use consalign::*;
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_consalign(criterion: &mut Criterion) {
  let input_file_path = Path::new(EXAMPLE_FASTA_FILE_PATH);
  let fasta_file_reader = Reader::from_file(Path::new(&input_file_path)).unwrap();
  let mut fasta_records = FastaRecords::new();
  let mut max_seq_len = 0;
  for fasta_record in fasta_file_reader.records() {
    let fasta_record = fasta_record.unwrap();
    let mut seq = convert(fasta_record.seq());
    seq.insert(0, PSEUDO_BASE);
    seq.push(PSEUDO_BASE);
    let seq_len = seq.len();
    if seq_len > max_seq_len {
      max_seq_len = seq_len;
    }
    fasta_records.push(FastaRecord::new(String::from(fasta_record.id()), seq));
  }
  let num_of_threads = num_cpus::get() as NumOfThreads;
  let mut thread_pool = Pool::new(num_of_threads);
  let output_dir_path = Path::new(OUTPUT_DIR_PATH);
  criterion.bench_function("wrapped_consalign::<u8, u8>", |b| {
    b.iter(|| {
      let _ = wrapped_consalign::<u8, u8>((
        &mut thread_pool,
        &fasta_records,
        output_dir_path,
        input_file_path,
        DEFAULT_MIN_BPP_ALIGN,
        DEFAULT_MIN_ALIGN_PROB_ALIGN,
        ScoringModel::Ensemble,
        TrainType::TrainedTransfer,
        true,
        DEFAULT_MIN_BPP_ALIGN_TURNER,
        DEFAULT_MIN_ALIGN_PROB_ALIGN_TURNER,
        false,
      ));
    });
  });
}

criterion_group!(benches, bench_consalign);
criterion_main!(benches);
