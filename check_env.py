import google.generativeai as genai

# Configure with your AI Studio API key
genai.configure(api_key="AIzaSyBcIy4IuRD_Ih-umLWKucjmkqLPVsQIi3Q")
#AIzaSyBRAKfYEnbImVZOEESX7KuIA8Op5mWI9js
#AIzaSyCIKtzRqs3qghcskPZpqLIrHnU8_nN-P7c
# AIzaSyDi1He5BVUSAjwYBd9in8errorDgZCqlYQ
# Create the model with the correct name
model = genai.GenerativeModel("gemini-2.5-flash-lite-preview-06-17")

# Call generate_content
response = model.generate_content("Summarize the main features of Gemini 2.5 Flash-Lite.")

# Print the response text
print(response.text)
