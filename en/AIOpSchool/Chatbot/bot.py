from Levenshtein import distance

class ChatBot:
    conversation = {}
    
    def __init__(self, name):
        self.name = name
        
    
    def train(self, conversation):
        self.conversation = conversation
        
    
    def get_response(self, question):
        if self.conversation == {}:
            return "Geen antwoord"
        
        question = question.lower()
        
        lowest_distance = 99999
        best_reply = "Geen antwoord"
        
        for known_question, reply in self.conversation.items():
            current_distance = distance(question, known_question.lower())
            current_distance = current_distance / max(len(known_question), len(question))
            if current_distance < lowest_distance:
                lowest_distance = current_distance
                best_reply = reply
        
        return best_reply