# GitHub Repository Setup for Humanoid Robotics Book

This document provides the exact steps you need to follow to deploy your book to GitHub Pages.

## Step 1: Create the GitHub repository manually

1. Go to https://github.com and log in to your account (Amnaahmed798)
2. Click the '+' icon in the top right corner and select 'New repository'
3. Set the repository name to 'humanoid-robotics-book'
4. Set the repository to 'Public' (required for GitHub Pages)
5. Do NOT initialize with README, .gitignore, or license (we already have these)
6. Click 'Create repository'

## Step 2: Add the remote origin and push the code

After creating the repository, copy the HTTPS clone URL (e.g., https://github.com/Amnaahmed798/humanoid-robotics-book.git)

Then run these commands in your terminal:

```bash
cd /home/misbah/projects/humanoid-robotics-book
git remote add origin https://github.com/Amnaahmed798/humanoid-robotics-book.git
git branch -M main
git push -u origin main
```

## Step 3: Configure GitHub Pages

1. Go to your repository on GitHub
2. Click on the 'Settings' tab
3. Scroll down to the 'Pages' section
4. Under 'Source', select 'Deploy from a branch'
5. Select 'main' branch and '/gh-pages' folder (or 'gh-pages' branch if using the deploy method)
6. Click 'Save'

## Step 4: Deploy the Docusaurus site

After pushing the code, run these commands to deploy to GitHub Pages:

```bash
cd /home/misbah/projects/humanoid-robotics-book/docusaurus
npm run deploy
```

This will build and deploy your Docusaurus site to GitHub Pages.

Your site will be available at: https://Amnaahmed798.github.io/humanoid-robotics-book/

## Alternative deployment method (if npm run deploy doesn't work):

You can also build and deploy manually:

```bash
cd /home/misbah/projects/humanoid-robotics-book/docusaurus
npm run build
cd build
git init
git add .
git commit -m 'Deploy to GitHub Pages'
git remote add origin https://github.com/Amnaahmed798/humanoid-robotics-book.git
git push -f origin main:gh-pages
```

## Note

The site configuration has been updated with your GitHub username (Amnaahmed798) and is ready for deployment once you complete these steps.