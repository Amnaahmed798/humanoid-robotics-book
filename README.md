# Physical AI & Humanoid Robotics Book

This repository contains a comprehensive educational resource covering the development of autonomous humanoid robots using NVIDIA Isaac, ROS 2, and AI technologies.

## Viewing the Book Locally

To view the book locally:

1. Navigate to the docusaurus directory:
   ```bash
   cd docusaurus
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run start
   ```

4. Open your browser to `http://localhost:3000/humanoid-robotics-book/`

## Deploying to GitHub Pages

To deploy the book to GitHub Pages:

1. Ensure your GitHub repository is set up and you have push access

2. Set the correct values in `docusaurus.config.js`:
   - `organizationName`: Your GitHub username or organization name
   - `projectName`: Your repository name (usually `your-username.github.io` or the repo name)
   - `url`: Your GitHub Pages URL (e.g., `https://your-username.github.io`)

3. Build the site:
   ```bash
   npm run build
   ```

4. Deploy to GitHub Pages:
   ```bash
   npm run deploy
   ```

Alternatively, you can set up GitHub Actions for automatic deployment on pushes to the main branch.

## Project Structure

- `book/` - Source content for the book organized by modules
- `code/` - Code examples referenced in the book
- `docusaurus/` - Docusaurus documentation site (what gets deployed)
- `specs/` - Project specifications and planning documents
- `history/` - Project history including ADRs and PHRs

## Book Modules

1. **Module 1**: The Robotic Nervous System (ROS 2)
2. **Module 2**: Digital Twin (Gazebo & Unity)
3. **Module 3**: The AI-Robot Brain (NVIDIA Isaac)
4. **Module 4**: Vision-Language-Action (VLA)

## About

This book covers over 4,000 lines of educational content, 50+ code examples with 95%+ success rate, complete ROS 2 architecture with Isaac integration, simulation-to-reality transfer techniques, and multi-modal interaction systems.