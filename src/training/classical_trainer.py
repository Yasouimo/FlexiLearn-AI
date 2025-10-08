def train_classical_model(model, X_train, y_train, X_test, y_test, problem_type):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if problem_type == "Classification":
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, y_pred)
        return {"model": model, "accuracy": accuracy, "y_pred": y_pred}
    else:
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"model": model, "mse": mse, "r2": r2}