import config

def answer_with_bidaf(passage, question):
    result=config.predictor.predict(passage=passage, question=question)
    
    return result['best_span_str']
