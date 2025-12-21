import os
import re
from vlmeval.smp import *

def extract_number_from_prediction(text):
    text = str(text).replace(',', '')
    all_numbers = re.findall(r'[+-]?\d+(?:\.\d+)?', text)
    
    return all_numbers[-1] if all_numbers else None

def extract_option_from_prediction(text):
    match = re.search(r'[\(\[A-Z]\s*答案是?[:：]?\s*\'?"?([A-Z])\'?"?', str(text), re.IGNORECASE)
    if match: return match.group(1)
    match = re.search(r'[\[\(\s,.]([A-Z])[\]\)\s,.]', f" {str(text)} ")
    if match: return match.group(1)
    match = re.match(r'\s*([A-Z])', str(text))
    if match: return match.group(1)
    return None


def calculate_metric_score_with_relative_error(prediction_text, answer_text, start=0.5, end=0.95, interval=0.05):
    """
    Calculate MRA score for numerical questions.
    """
    pred_num_str = extract_number_from_prediction(prediction_text)

    if pred_num_str is None:
        return 0.0

    try:
        pred = float(pred_num_str)
        ans = float(answer_text)
    except (ValueError, TypeError):
        return 0.0

    if ans == 0:
        return 1.0 if pred == 0 else 0.0

    relative_error = abs(pred - ans) / abs(ans)

    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))

    error_tolerances = 1.0 - conf_intervs

    passed_checks = relative_error <= error_tolerances

    score = np.mean(passed_checks)

    print(f"Prediction: {pred}, Answer: {ans}, Relative Error: {relative_error:.4f}, Score: {score:.4f}")
    return score



# GT<0.60, we take it as stationary. to calculate MRA, we use 0.30 as GT when prediction>=0.30, otherwise we take it as correct (score=1) if prediction<0.30
# speed and displacement, Thres=0.30; traj_length, Thres=2.0
def calculate_metric_score_with_relative_error_consider_zero(
    prediction_text, answer_text, start=0.5, end=0.95, interval=0.05, Thres=0.30
):
    """
    Calculate MRA score for numerical questions, with special handling for GT==0.
    If GT==0, return 1.0 if prediction < Thres, else compute MRA with GT=Thres.
    """
    pred_num_str = extract_number_from_prediction(prediction_text)
    if pred_num_str is None:
        return 0.0

    try:
        pred = float(pred_num_str)
        ans = float(answer_text)
    except (ValueError, TypeError):
        return 0.0

    if ans == 0:
        if pred < Thres:
            return 1.0
        else:
            ans = Thres  # Use Thres as GT for MRA calculation

    relative_error = abs(pred - ans) / abs(ans)

    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    error_tolerances = 1.0 - conf_intervs
    passed_checks = relative_error <= error_tolerances
    score = np.mean(passed_checks)

    print(f"Prediction: {pred}, Answer: {ans}, Relative Error: {relative_error:.4f}, Score: {score:.4f}")
    return score
