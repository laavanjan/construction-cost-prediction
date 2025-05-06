from dataloader import generate_sample_data
from model import train_model

def main():
    # Step 1: Generate synthetic data (optional — you can skip if data already exists)
    print("Generating sample construction data...")
    generate_sample_data(n_samples=1000, output_file='construction_data.csv')

    # Step 2: Train the model
    print("Starting training process...")
    model = train_model(data_path='construction_data.csv', model_path='cost_model.pkl')

    print("Training completed successfully.")

if __name__ == "__main__":
    main()
