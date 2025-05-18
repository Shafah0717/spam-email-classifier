from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

emails = [
    "Congratulations! You won a $1000 Walmart gift card. Click here to claim.",
    "Hi John, can we meet tomorrow at 10 AM for the project discussion?",
    "Get cheap meds without prescription. Limited time offer!",
    "Dear user, your account has been compromised. Reset your password now.",
    "Let's catch up over coffee this weekend.",
    "You’ve been selected for a free cruise to the Bahamas! Call now!",
    "Reminder: Your doctor’s appointment is scheduled for Friday at 3 PM.",
    "Earn $5000 a week from home. Ask me how!",
    "Lunch tomorrow? Bring your laptop so we can work together.",
    "Lowest insurance rates guaranteed. Don’t miss this offer!"
]

labels = [
    "spam",        
    "not spam",    
    "spam",        
    "spam",        
    "not spam",    
    "spam",        
    "not spam",    
    "spam",        
    "not spam",    
    "spam"         
]

vectorize = CountVectorizer()
x = vectorize.fit_transform(emails)

x_train,x_test,y_train,y_test = train_test_split(x,labels,test_size=0.2,random_state=48)

model = MultinomialNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


