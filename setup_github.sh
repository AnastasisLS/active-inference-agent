#!/bin/bash

# Script to help set up GitHub repository for Active Inference Agent project

echo "=== Active Inference Agent Project - GitHub Setup ==="
echo ""
echo "This script will help you set up your GitHub repository."
echo ""

# Get repository name
read -p "Enter your GitHub username: " GITHUB_USERNAME
read -p "Enter repository name (default: active-inference-agent): " REPO_NAME
REPO_NAME=${REPO_NAME:-active-inference-agent}

echo ""
echo "Repository will be created as: $GITHUB_USERNAME/$REPO_NAME"
echo ""

# Instructions for manual creation
echo "=== Manual GitHub Repository Creation ==="
echo ""
echo "1. Go to https://github.com/new"
echo "2. Repository name: $REPO_NAME"
echo "3. Description: Active Inference Agent in a Simulated Environment - Week 1 Implementation"
echo "4. Make it Public or Private (your choice)"
echo "5. Do NOT initialize with README, .gitignore, or license (we already have these)"
echo "6. Click 'Create repository'"
echo ""
echo "After creating the repository, run these commands:"
echo ""
echo "git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo "git branch -M main"
echo "git push -u origin main"
echo ""

# Ask if user wants to proceed with remote setup
read -p "Have you created the repository? (y/n): " PROCEED

if [[ $PROCEED == "y" || $PROCEED == "Y" ]]; then
    echo ""
    echo "Setting up remote repository..."
    
    # Add remote origin
    git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git
    
    # Set branch to main
    git branch -M main
    
    # Push to GitHub
    echo "Pushing to GitHub..."
    git push -u origin main
    
    echo ""
    echo "=== Success! ==="
    echo "Your repository is now available at:"
    echo "https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    echo ""
    echo "You can now:"
    echo "1. View your code on GitHub"
    echo "2. Share the repository with others"
    echo "3. Continue development and push updates"
    echo ""
else
    echo ""
    echo "Please create the repository first, then run this script again."
    echo "Or run the git commands manually:"
    echo "git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    echo "git branch -M main"
    echo "git push -u origin main"
fi 