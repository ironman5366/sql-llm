"""Run large-scale experiments with corrected eval (no row caps anywhere)."""

import traceback
from research_harness import Experiment, load_csv_dataset

def run_one(gpu, port, csv_path, n_rows, train_budget, table_name):
    """Run a single insert-all + eval-all experiment."""
    dataset = load_csv_dataset(csv_path, table_name=table_name, n_rows=n_rows)
    print(f"\n{'='*70}")
    print(f"  {table_name}: {len(dataset['rows'])} rows, "
          f"{len(dataset['columns'])} cols, budget={train_budget}s, GPU {gpu}")
    print(f"{'='*70}\n")

    with Experiment(gpu=gpu, port=port, train_budget=train_budget) as exp:
        exp.create_and_insert(dataset)
        result = exp.eval_recall(dataset,
            query_types=["point_lookup", "range", "multi_condition", "full_scan"])
        print(f"\n{result.summary()}")

        # Print multi-condition details
        for d in result.query_details:
            if d.get("type", "").startswith("multi_condition"):
                print(f"  {d['type']}: {d.get('filters', '')} -> "
                      f"{d.get('status', d.get('got_count', '?'))}")
        return result


if __name__ == "__main__":
    import sys
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8100

    experiments = [
        # (csv_path, n_rows, train_budget, table_name)
        ("datasets/kaggle/country_bp_summary.csv", 86, 300, "bp_summary"),
        ("datasets/kaggle/blood_pressure_global_dataset.csv", 200, 300, "blood_pressure"),
        ("datasets/kaggle/blood_pressure_global_dataset.csv", 500, 300, "bp_500"),
        ("datasets/kaggle/blood_pressure_global_dataset.csv", 1000, 300, "bp_1000"),
        ("datasets/kaggle/blood_pressure_global_dataset.csv", 5000, 300, "bp_5000"),
    ]

    results = []
    for csv_path, n_rows, budget, name in experiments:
        try:
            r = run_one(gpu, port, csv_path, n_rows, budget, name)
            results.append((name, n_rows, r.recall_pct, r.correct_pct,
                            r.rows_recalled, r.total_rows))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            traceback.print_exc()
            results.append((name, n_rows, -1, -1, -1, n_rows))

    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS (GPU {gpu})")
    print(f"{'='*70}")
    for name, n_rows, recall, correct, recalled, total in results:
        print(f"  {name:20s} {n_rows:5d} rows | "
              f"recall={recall:5.1f}% ({recalled}/{total}) | "
              f"correct={correct:5.1f}%")
