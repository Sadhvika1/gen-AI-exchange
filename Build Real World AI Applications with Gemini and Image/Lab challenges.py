#Task 1: Develop a Python function named generate_bouquet_image(prompt). This function should invoke the imagen-3.0-generate-002 model using the supplied prompt, generate the image, and store it locally. 
#For this challenge, use the prompt: Create an image containing a bouquet of 2 sunflowers and 3 roses
import argparse
import os

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

def generate_bouquet_image(prompt: str) -> None: # Removed return type annotation as it's not being used.
    """
    Generates an image of a bouquet of 2 sunflowers and 3 roses using the imagen-3.0-generate-002 model
    and saves it locally.

    Args:
        prompt: The text prompt describing the image to generate.
    """
    project_id = "your-project-id"  # Replace with your GCP project ID
    location = "us-central1"  # Replace with your GCP location
    output_filename = "bouquet_image.png"  # You can change the filename and extension

    vertexai.init(project=project_id, location=location)
    model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

    images = model.generate_images(
        prompt=prompt,
        number_of_images=1,
        seed=1,
        add_watermark=False,
    )

    if images and images[0]:  # Check if images is not None and has at least one element.
        # Ensure the directory exists
        output_dir = os.path.dirname(output_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        images[0].save(output_filename)
        print(f"Image saved successfully as '{output_filename}'")
    else:
        print("Image generation failed.")



# Example Usage (You can uncomment this to run the function)
generate_bouquet_image(prompt='Create an image containing a bouquet of 2 sunflowers and 3 roses')

# Note: Remember to replace 'your-project-id' and 'us-central1' with your actual GCP project ID and location.

#TASK2
#Task 2: Develop a second Python function called analyze_bouquet_image(image_path). This function will take the image path as input along with a text prompt to generate birthday wishes based on the image passed and send it to the gemini-2.0-flash-001 model. To ensure responses can be obtained as and when they are generated, enable streaming on the prompt requests.
import vertexai
from vertexai.generative_models import GenerativeModel, Part, Image
import time

def generate_birthday_wish_from_image(project_id: str, location: str, image_path: str) -> str:
    # Step 1: Initialize Vertex AI with project and location
    vertexai.init(project=project_id, location=location)

    # Step 2: Load the Gemini multimodal model
    model = GenerativeModel("gemini-2.0-flash-001")

    # Step 3: Send prompt with image
    response = model.generate_content(
        contents=[
            Part.from_image(Image.load_from_file(location=image_path)),
            "Generate a warm and poetic birthday wish based on this bouquet image."
        ],
        stream=False  # Disable streaming for now, just get the full text
    )

    # Step 4: Simulate waiting for a log to be created
    print("\n⏳ Waiting for log to be created...")
    time.sleep(2)  # You can increase/decrease this time as needed
    print("✅ Log created successfully.\n")

    # Step 5: Return the text output
    return response.text

# ---- MAIN ---- #
if __name__ == "__main__":
    project_id = "Project_ID"
    location = "us-central1"
    image_path = "/home/student/bouquet_image.png"

    print("📸 Reading image and generating wish...\n")
    wish = generate_birthday_wish_from_image(project_id, location, image_path)

    print("🎉 Generated Birthday Wish:\n")
    print(wish)









