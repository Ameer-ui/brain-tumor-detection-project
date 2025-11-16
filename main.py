from data_loader import load_data
from test_model_training import train_model
from test_prediction import predict_and_visualize

if __name__ == "__main__":
    train_loader, test_loader, class_names = load_data()
    model = train_model(train_loader, test_loader)
    predict_and_visualize(model, test_loader)