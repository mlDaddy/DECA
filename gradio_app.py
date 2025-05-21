import gradio as gr
import os
import time
from PIL import Image
import datetime
import tkinter as tk
from tkinter import filedialog

class ImageGeneratorGUI:
    def __init__(self, api_handler):
        self.api_handler = api_handler
        self.DALLE3_RESOLUTIONS = ["1024x1024", "1792x1024", "1024x1792"]
        self.GPT_IMAGE_RESOLUTIONS = ["1024x1024", "1536x1024", "1024x1536", "auto"]
        self.DALLE2_RESOLUTIONS = ["256x256", "512x512", "1024x1024"]
        # Initialize output directory
        self.output_dir = os.path.join(os.getcwd(), "generated_images")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def create_interface(self):
        with gr.Blocks(title="OpenAI Image Generator") as app:
            gr.Markdown("# OpenAI Image Generator")

            # Add a notice about GPT-Image-1 verification
            gr.Markdown("""
            > **Note:** Using GPT-Image-1 requires organization verification.
            > If you encounter verification errors, please go to
            > [OpenAI Organization Settings](https://platform.openai.com/settings/organization/general)
            > and click on Verify Organization.
            """)

            # Single column layout
            api_key = gr.Textbox(label="OpenAI API Key", type="password")

            # Mode selection including test mode as an option
            mode = gr.Radio(
                ["text2img", "img2img", "test mode (for gui testing)"],
                label="Mode",
                value="text2img"
            )

            # Model selection
            model_dropdown = gr.Dropdown(
                choices=["DALL-E-2", "DALL-E-3", "GPT-Image-1"],
                label="Model",
                value="DALL-E-2"
            )

            resolution = gr.Dropdown(
                choices=self.DALLE2_RESOLUTIONS,
                label="Resolution",
                value=self.DALLE2_RESOLUTIONS[0]
            )

            prompt = gr.Textbox(label="Prompt", lines=3)

            # Input image only visible for img2img mode
            input_image = gr.Image(label="Input Image (for Image-to-Image)", type="pil", visible=False)

            # Test image for test mode
            test_image_input = gr.Image(label="Test Image (for Test Mode)", type="pil", visible=False)

            generate_btn = gr.Button("Generate Image")
            progress = gr.Slider(0, 100, value=0, label="Progress")

            # Status message area
            status_msg = gr.Textbox(label="Status", interactive=False)

            # Output image
            output_image = gr.Image(label="Generated Image", type="pil")

            # Output folder selection
            with gr.Row():
                output_folder = gr.Textbox(
                    label="Output Folder",
                    value=self.output_dir,
                    interactive=True
                )
                browse_folder_btn = gr.Button("Browse...")

            # Filename input
            filename_input = gr.Textbox(
                label="Filename (without extension)",
                value=f"generated_image_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            # Save options
            with gr.Row():
                save_btn = gr.Button("Save Image")
                download_btn = gr.Button("Download Image", visible=True)

            # File download component
            download_file = gr.File(label="Download Image", visible=False)

            # Event handlers
            model_dropdown.change(
                self.update_resolution_options,
                inputs=model_dropdown,
                outputs=resolution
            )

            mode.change(
                self.update_visibility,
                inputs=mode,
                outputs=[model_dropdown, input_image, test_image_input]
            )

            generate_btn.click(
                self.Image_Generation,
                inputs=[api_key, mode, model_dropdown, resolution, prompt, input_image, test_image_input],
                outputs=[progress, output_image, status_msg, filename_input]
            )

            browse_folder_btn.click(
                self.browse_output_folder,
                inputs=None,
                outputs=output_folder
            )

            save_btn.click(
                self.save_image_to_folder,
                inputs=[output_image, output_folder, filename_input],
                outputs=status_msg
            )

            download_btn.click(
                self.prepare_download,
                inputs=[output_image, filename_input],
                outputs=[download_file, status_msg]
            )

        return app

    def update_resolution_options(self, model):
        if model == "DALL-E-3":
            return gr.update(choices=self.DALLE3_RESOLUTIONS, value=self.DALLE3_RESOLUTIONS[0])
        elif model == "DALL-E-2":
            return gr.update(choices=self.DALLE2_RESOLUTIONS, value=self.DALLE2_RESOLUTIONS[2])
        else:  # GPT-Image-1
            return gr.update(choices=self.GPT_IMAGE_RESOLUTIONS, value=self.GPT_IMAGE_RESOLUTIONS[0])

    def update_visibility(self, mode):
        # Update visibility based on mode selection
        if mode == "text2img":
            # For text2img, show all models
            model_choices = ["DALL-E-2", "DALL-E-3", "GPT-Image-1"]
            model_value = "DALL-E-2"
            input_image_visible = False
            test_image_visible = False
        elif mode == "img2img":
            # For img2img, only GPT-Image-1 is valid
            model_choices = ["DALL-E-2", "GPT-Image-1"]
            model_value = "DALL-E-2"
            input_image_visible = True
            test_image_visible = False
        else:  # test mode
            model_choices = ["DALL-E-2", "DALL-E-3", "GPT-Image-1"]
            model_value = "DALL-E-2"
            input_image_visible = False
            test_image_visible = True

        return [
            gr.update(choices=model_choices, value=model_value),
            gr.update(visible=input_image_visible),
            gr.update(visible=test_image_visible)
        ]

    def browse_output_folder(self):
        try:
            # Create and hide root window
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            # Open folder selection dialog
            folder_path = filedialog.askdirectory(
                title="Select Output Folder",
                initialdir=self.output_dir
            )

            root.destroy()

            if folder_path:
                self.output_dir = folder_path
                return folder_path
            else:
                return self.output_dir
        except Exception as e:
            print(f"Error in folder selection: {e}")
            return self.output_dir

    def Image_Generation(self, api_key, mode, model, resolution, prompt, input_image=None, test_image=None):
        is_test_mode = mode.startswith("test mode")

        # Generate a new filename based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"generated_image_{timestamp}"

        if is_test_mode:
            # Simulate API call with delay
            for i in range(10):
                time.sleep(0.5)
                yield i*10, test_image if test_image else None, "Test mode: Processing...", new_filename

            # Return test image or a placeholder
            if test_image:
                yield 100, test_image, "Test mode: Completed with test image", new_filename
            else:
                # Create a simple placeholder image with text
                img = Image.new('RGB', (512, 512), color=(73, 109, 137))
                yield 100, img, "Test mode: Completed with placeholder image", new_filename
            return

        try:
            # Set API key for handler
            self.api_handler.api_key = api_key
            self.api_handler.headers["Authorization"] = f"Bearer {api_key}"

            # Initialize progress
            yield 10, None, "Starting image generation...", new_filename

            if mode == "text2img":
                # Text to image generation
                yield 30, None, f"Generating image with {model}...", new_filename
                image = self.api_handler.text_to_image(model, resolution, prompt)
                yield 70, None, "Processing generated image...", new_filename
                yield 100, image, f"Image successfully generated with {model}", new_filename

            elif mode == "img2img":
                # Image to image editing
                yield 30, None, "Processing input image...", new_filename
                yield 50, None, "Applying edits to image...", new_filename

                try:
                    model_used = model
                    image = self.api_handler.image_to_image(model, resolution, prompt, input_image)
                except ValueError as e:
                    if "verified" in str(e).lower():
                        # If verification error, try with DALL-E-2 fallback
                        yield 60, None, "GPT-Image-1 requires verification. Falling back to DALL-E-2...", new_filename
                        image = self.api_handler._fallback_dalle2_edit(resolution, prompt, input_image)
                        model_used = "DALL-E-2 (fallback)"
                    else:
                        raise

                yield 70, None, "Finalizing image...", new_filename
                yield 100, image, f"Image successfully edited with {model_used}", new_filename

        except Exception as e:
            yield 100, None, f"Error: {str(e)}", new_filename
            raise gr.Error(f"Error: {str(e)}")

    def save_image_to_folder(self, img, folder_path, filename):
        if img is None:
            return "No image to save"

        try:
            # Ensure folder exists
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # Clean filename and add extension
            filename = os.path.basename(filename)
            if not filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_image_{timestamp}"

            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename += ".png"

            # Full path to save
            full_path = os.path.join(folder_path, filename)

            # Save the image
            img.save(full_path)
            return f"Image saved to {full_path}"
        except Exception as e:
            return f"Error saving image: {str(e)}"

    def prepare_download(self, img, filename):
        if img is None:
            return gr.update(visible=False), "No image to save"

        try:
            # Clean filename and add extension
            filename = os.path.basename(filename)
            if not filename:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_image_{timestamp}"

            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename += ".png"

            # Save the image to a temporary file
            temp_path = os.path.join(self.output_dir, filename)
            img.save(temp_path)

            # Make the download component visible and return the file path
            return gr.update(value=temp_path, visible=True), f"Click the download button above to save {filename}"
        except Exception as e:
            return gr.update(visible=False), f"Error preparing download: {str(e)}"