import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

# Load the image captioning model and tokenizer
caption_model_name = "Salesforce/blip-image-captioning-large"
caption_processor = BlipProcessor.from_pretrained(caption_model_name)
caption_model = BlipForConditionalGeneration.from_pretrained(caption_model_name)

def generate_caption_and_analyze_emotions(image):
    # Preprocess the image for caption generation
    caption_inputs = caption_processor(images=image, return_tensors="pt")

    # Generate caption using the caption model
    caption = caption_model.generate(**caption_inputs)

    # Decode the output caption
    decoded_caption = caption_processor.decode(caption[0], skip_special_tokens=True)

    # Load the emotion analysis model and tokenizer
    emotion_model_name = "SamLowe/roberta-base-go_emotions"
    emotion_classifier = pipeline(model=emotion_model_name)

    results = emotion_classifier(decoded_caption)
    sentiment_label = results[0]['label']
    if sentiment_label == 'neutral':
        sentiment_text = "Sentiment of the image is"
    else:
        sentiment_text = "Sentiment of the image shows"

    final_output = f"This image shows {decoded_caption} and {sentiment_text} {sentiment_label}."

    return final_output

# Define the Gradio interface
inputs = gr.inputs.Image(label="Upload an image")
outputs = gr.outputs.Textbox(label="Sentiment Analysis")

# Create the Gradio app
app = gr.Interface(fn=generate_caption_and_analyze_emotions, inputs=inputs, outputs=outputs)

# Launch the Gradio app
if __name__ == "__main__":
    app.launch()
