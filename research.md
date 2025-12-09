# Research Findings: Physical AI & Humanoid Robotics Book

## APA Citation Style Best Practices

### In-Text Citations
- For direct quotes: (Author, Year, p. #)
- For paraphrasing: (Author, Year)
- For multiple authors: (Author1 et al., Year) for 3+ authors
- For multiple sources: (Author1, Year; Author2, Year)

### Reference List Format
- Alphabetical by author's last name
- Hanging indent for each reference
- Title case for article titles, sentence case for journal titles

### Common Formats
- Journal article: Author, A. A. (Year). Title of article. *Title of Periodical*, volume(issue), pages. https://doi.org/xx.xxx/yyyy
- Book: Author, A. A. (Year). *Title of work*. Publisher.
- Website: Author, A. A. (Year, Month Date). Title of webpage. Site Name. URL

## Docusaurus Configuration for Multi-Chapter Books

### Directory Structure
- `/docs/` for main documentation
- `/docs/module1/`, `/docs/module2/`, etc. for modules
- `/docs/appendices/` for appendices
- `/docs/references/` for references

### Code Block Highlighting
- Docusaurus uses Prism.js by default
- Supports many languages including Python, C++, etc.
- Can be customized with themes

### Navigation
- Sidebars.js controls the navigation structure
- Categories can be collapsible
- Can have multiple sidebars for different sections

## GitHub Pages Deployment Workflows

### GitHub Actions
- Use GitHub Actions for automated deployment
- Workflow file typically in `.github/workflows/deploy.yml`
- Can be configured to deploy on push to main branch
- Uses `peaceiris/actions-gh-pages` or similar action

### Configuration Settings
- `baseUrl` in docusaurus.config.js should match project name for GitHub Pages
- `organizationName` and `projectName` should match repository details
- Output builds to `gh-pages` branch by default

## Strategies for Code Example Reproducibility

### Environment Setup
- Provide detailed environment setup instructions
- Use Docker containers for consistent environments
- Specify exact versions of dependencies
- Include requirements.txt or similar files

### Testing Framework
- Create automated tests for code examples
- Test in sandboxed environments
- Verify expected outputs
- Document common troubleshooting steps

### Documentation Standards
- Include expected output for each example
- Provide error handling guidance
- Document system requirements
- Include version compatibility information