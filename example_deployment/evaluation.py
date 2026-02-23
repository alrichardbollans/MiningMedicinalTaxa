import pandas as pd


def main():
    manual_checks = pd.read_csv('manual_outputs.csv')
    correct = manual_checks[manual_checks['decision'] == 'Yes']
    incorrect = manual_checks[manual_checks['decision'] == 'No']
    print(f"Correct: {len(correct)}")
    print(f"Incorrect: {len(incorrect)}")

    precision = len(correct)/(len(correct)+len(incorrect))
    print(f"Precision: {precision}")


if __name__ == '__main__':
    main()
