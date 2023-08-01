**App Link:** https://huggingface.co/spaces/muneebashraf/Visual-Sentiment-Analyzer

**Visual-Sentiment-Analyzer:**
Visual sentiment analyzer is deployed on the hugging face. It provides details and detects the sentiment described in the image. For this, I'm using two hugging face models. The first generates a caption of the provided image, and the second model takes that caption as input to show the emotion described in that text. 

 **Hugging Face Models Used**
**Blip:** For generating caption of the provided image.
**Roberta-base-go_emotions:** For generating sentiment from the caption

**Reference**
https://huggingface.co/Salesforce/blip-image-captioning-large 
print('/n')
https://huggingface.co/SamLowe/roberta-base-go_emotions
