from flask import Flask, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
app = Flask(__name__)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def predict_box_score(historical_data, future_box_info):
    try:
        prompt = f"""You’re an expert in predicting Goodiebox satisfaction, skilled at simulating outcomes from historical trends. Using this data:
        Historical Data: {historical_data}
        Future Box Info: {future_box_info}
        Predict a satisfaction score (1–5) for the future box based on trends in past ratings, product variety, and value. Return the score as a number with two decimal places (e.g., 4.23), nothing else."""
        response = client.chat.completions.create(
            model="o1-mini",
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=100
        )
        score = response.choices[0].message.content.strip()
        logger.info(f"Raw model response: '{score}'")
        if not score:
            logger.error("Model returned an empty response")
            raise ValueError("Model returned an empty response")
        try:
            score_float = float(score)
            if not (1 <= score_float <= 5):
                raise ValueError("Score out of range")
            score = f"{score_float:.2f}"
        except ValueError as e:
            logger.error(f"Invalid score format received: '{score}', error: {str(e)}")
            raise ValueError(f"Invalid score format received: '{score}'")
        return score
    except Exception as e:
        logger.error(f"Error in box score simulation: {str(e)}")
        raise Exception(f"Error in box score simulation: {str(e)}")

@app.route('/predict_box_score', methods=['POST'])
def box_score():
    try:
        data = request.get_json()
        if not data or 'future_box_info' not in data:
            logger.warning("Missing future box info in request")
            return jsonify({'error': 'Missing future box info'}), 400
        historical_data = data.get('historical_data', 'No historical data provided')
        future_box_info = data['future_box_info']
        score = predict_box_score(historical_data, future_box_info)
        return jsonify({'predicted_box_score': score})
    except Exception as e:
        logger.error(f"Error in /predict_box_score endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

