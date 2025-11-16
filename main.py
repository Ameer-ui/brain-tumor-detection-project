from data_loader import load_data
from test_model_training import train_model
from test_prediction import predict_and_visualize

if __name__ == "__main__":
    train_gen, test_gen, _ = load_data()
    model = train_model(train_gen, test_gen)
    predict_and_visualize(model, test_gen)
