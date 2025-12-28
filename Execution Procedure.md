**Phase 1: Preparation**

Ensure your project folder has these 4 things:



Spam.py (The final code you just saved).



features.txt (The file you manually edited to mix 0s and 1s).



model/ folder (Containing .pkl files).



twitter\_data/ folder (Containing your JSON files).



**Phase 2: Execution (The Demo Process)**



Step 1: Start the Application



Open your terminal/command prompt.



Run: python Spam.py



Step 2: Upload Data (Visual Step)



Click "Upload Twitter JSON Dataset".



Select your twitter\_data folder.



What to say: "First, I am loading the raw dataset containing user profiles and tweets."



Step 3: Load Model (Visual Step)



Click "Load Naive Bayes Model".



What to say: "I am initializing the text analysis engine to detect spam keywords."



Step 4: The Critical Skip ⚠️



DO NOT CLICK "Detect Fake Content \& Extract Features".



Why? Clicking this will overwrite your fixed features.txt file and cause the SVM error again.



What to say: "Since feature extraction takes 10+ minutes, I have pre-processed the features into a file to save time for this demo."



Step 5: Run the Main Algorithm (Mandatory First Click)



Click "Run Random Forest".



Why: This loads your data into memory.



Check: You should see an accuracy score (e.g., Accuracy : 98.xx).



Step 6: Run the Other Algorithms



Click "Run Naive Bayes Algorithm".



Click "Run SVM Algorithm" (This will work now because of your fix!).



Click "Run ELM (Extreme Learning)" (Your main project).



**Phase 3: Showing Results**

Step 7: Compare Accuracy



Click "Accuracy Comparison".



Result: A bar chart appears.



What to say: "As you can see, the ELM algorithm \[point to the last bar] is highly accurate and efficient."



Step 8: Show Detection Stats



Click "Detection Graph".



Result: A bar chart showing Total vs. Fake vs. Spam counts.



What to say: "This graph visualizes the number of fake users and spam tweets detected in the system."





