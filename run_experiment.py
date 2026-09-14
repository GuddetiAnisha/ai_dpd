from src.experiment import run

if __name__ == "__main__":
    df = run("outputs")
    print("\nExperiment completed.\n")
    print(df.to_string(index=False))
    print("\nSaved results in ./outputs")
