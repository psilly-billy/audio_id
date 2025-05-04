# Secure Deployment Guide

This guide explains the secure handling of authentication tokens in the Video Accent Analyzer application.

## Hugging Face Token Handling

The application requires a Hugging Face token to download the Distil-Whisper model. Here's how to securely set up this token:

### Local Development

1. **Environment File (Recommended)**
   - Copy `dotenv.example` to `.env`
   - Add your token: `HF_TOKEN=your_token_here`
   - This file is excluded from git via `.gitignore`

2. **Environment Variable**
   - Set directly in your terminal: `export HF_TOKEN=your_token_here`
   - Windows: `set HF_TOKEN=your_token_here`

3. **Config File (Legacy Method)**
   - Create `config.py` with: `HF_TOKEN = "your_token_here"`
   - This file is excluded from git via `.gitignore`

### Streamlit Cloud Deployment

1. **Streamlit Secrets**
   - Use the Streamlit Cloud dashboard
   - Go to App Settings > Secrets
   - Add your token as `HF_TOKEN = "your_token_here"`

2. **Secrets File Format**
   - For reference, see `.streamlit/secrets.toml.example`
   - Never commit actual secrets to the repository

### GitHub Deployment

1. **GitHub Secrets**
   - If using GitHub Actions, set up secrets in your repository settings
   - Go to Settings > Secrets and Variables > Actions
   - Add a new repository secret named `HF_TOKEN`

2. **GitHub Actions Workflow**
   - Use secrets in workflows with: `${{ secrets.HF_TOKEN }}`
   - Example workflow snippet:
     ```yaml
     jobs:
       deploy:
         steps:
           - name: Run tests
             env:
               HF_TOKEN: ${{ secrets.HF_TOKEN }}
             run: python run_tests.py
     ```

## Security Best Practices

1. **Never commit tokens or credentials** to your repository
2. **Never hardcode tokens** in your application code
3. **Rotate tokens periodically** for enhanced security
4. **Use minimal scopes/permissions** for your tokens
5. **Monitor token usage** in your Hugging Face account

## Troubleshooting

If you encounter authentication errors:

1. Verify your token is valid in the Hugging Face dashboard
2. Check that the token is being properly loaded by the application
3. Try the manual token entry option in the UI for testing (not recommended for production)
4. Check the application logs for authentication errors 