# OmniGen TCP Server Setup

This document outlines the steps to set up the OmniGen TCP Server environment and download necessary models. This code is based on the OmniGen model: [OmniGen Repository](https://github.com/VectorSpaceLab/OmniGen).

## Prerequisites

Ensure you have `apt`, `python`, and `wget` installed on your system.

## Installation Steps

Follow the steps below to set up your environment:

1. **Update package lists**:

   ```bash
   sudo apt update
   ```

2. **Install net-tools**:

   ```bash
   sudo apt install net-tools
   ```

3. **Create a Python virtual environment**:

   ```bash
   python -m venv env
   ```

4. **Activate the virtual environment**:

   ```bash
   source env/bin/activate
   ```

5. **Create a directory for models**:

   ```bash
   mkdir Models
   ```

6. **Clone the OmniGen TCP server repository**:

   ```bash
   git clone https://github.com/Ordoumpozanis/Omnigen-tcp-server
   ```

7. **Change into the Models directory**:

   ```bash
   cd Models
   ```

8. **Create a script file named `download_models.sh`**:

   ```bash
   touch download_models.sh
   ```

9. **Add the following code to `download_models.sh`**:

   ```bash
   #!/bin/bash

   # List of URLs to download
   urls=(
       "https://huggingface.co/Shitao/OmniGen-v1/resolve/main/.gitattributes"
       "https://huggingface.co/Shitao/OmniGen-v1/resolve/main/README.md"
       "https://huggingface.co/Shitao/OmniGen-v1/resolve/main/config.json"
       "https://huggingface.co/Shitao/OmniGen-v1/resolve/main/demo_cases.png"
       "https://huggingface.co/Shitao/OmniGen-v1/resolve/main/model.safetensors"
       "https://huggingface.co/Shitao/OmniGen-v1/resolve/main/special_tokens_map.json"
       "https://huggingface.co/Shitao/OmniGen-v1/resolve/main/tokenizer.json"
       "https://huggingface.co/Shitao/OmniGen-v1/resolve/main/tokenizer_config.json"
   )

   # Loop through each URL and use wget to download
   for url in "${urls[@]}"; do
       wget "$url"
   done
   ```

10. **Make the script executable**:

    ```bash
    chmod +x download_models.sh
    ```

11. **Run the script to download the models**:

    ```bash
    ./download_models.sh
    ```

12. **Navigate to the OmniGen folder**:

    ```bash
    cd OmniGen
    ```

13. **Install the required Python packages**:

    ```bash
    pip install -r requirements.txt
    ```

14. **Configure Accelerate**:

    ```bash
    accelerate config
    ```

15. **Launch the server**:
    ```bash
    PORT=7860 HOST=0.0.0.0 accelerate launch server.py
    ```

## Important Note

- Ensure to expose port **7860** for TCP to allow access to the server.

## Notes

- Make sure to have appropriate permissions and internet access for downloading files.
- Adjust the `PORT` and `HOST` settings as needed for your environment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
