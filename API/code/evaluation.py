import pandas as pd
import joblib
import shap
from interpret import show, show_link


EXPORT_PATH = '../exportNew/'

def evaluate():

    model_activity = joblib.load(f'{EXPORT_PATH}/model_activity.pkl')
    model_usage = joblib.load(f'{EXPORT_PATH}/model_usage.pkl')

    X_train_usage = pd.read_pickle(f'{EXPORT_PATH}/X_train_usage.pkl')

    X_test_activity = pd.read_pickle(f'{EXPORT_PATH}/X_test_activity.pkl')
    X_test_usage = pd.read_pickle(f'{EXPORT_PATH}/X_test_usage.pkl')
    y_test_activity = pd.read_pickle(f'{EXPORT_PATH}/y_test_activity.pkl')

    ebm_global_activity = model_activity.explain_global()
    ebm_local_activity = model_activity.explain_local(X_test_activity, y_test_activity)
    ebm_global_activity_url = (show_link(ebm_global_activity))
    ebm_local_activity_url = (show_link(ebm_local_activity))

    X_train_summary = shap.sample(X_train_usage, 100)
    explainer = shap.KernelExplainer(model_usage.predict_proba, X_train_summary)
    base_value = explainer.expected_value[1]
    shap_values = explainer.shap_values(X_test_usage, check_additivity=False)
    usage_plot = shap.force_plot(base_value, shap_values[1], X_test_usage)

    shap.save_html('static/usage_shap_plot.html', usage_plot)

    evaluations = {
            'global': ebm_global_activity_url,
            'local': ebm_local_activity_url
    }

    return evaluations

if __name__ == '__main__':
    evaluate()
