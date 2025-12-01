# Hackathon 2025 Round 2 - Web App Template

This repository serves as a starter template for building your optimization web application for Hackathon 2025 Round 2. It includes a basic structure, example code, and documentation to help you get started quickly.


##  Quick Start

1. **Use this template** - Click "Use this template" button to create your team's repository
2. **Clone** your new repository locally
3. **Build your web application** in your cloned repository
    - Preferred techstack as below: 
        - Python streamlit
        - Python FastAPI + React
        - Node.js + React
5. **Test locally** and ensure everything works
6. **Push your code** to your cloned repository 
7. **Update the README.md** with your project details and setup instructions. 
7. **Create your project documentation** using the provided mkdocs template
8. **Round2 presentation** - Present your submission using the local mkdocs server and local webapp from your system during the presentation day.(9th/10th Oct 2025)


##  Important Notes

1. **Source Code**: All code must be within your team's repository. 
2. **Local Execution**: Add the setup instructions with README.md, ensure your app runs locally across Windows/Mac/Linux platforms. No separate web hosting needed.
3. **Model source code**:
    - 3.1 Add  your Round 1 ML models (volume, elasticity, powder) scripts within 'src' folder. The predictions from this round1 model should __exactly match with that submitted during round1__.
    - 3.2 Add your optimizer scripts within 'src' folder. You can modify these scripts as needed to improve optimization performance.
4. **Real-time Updates**: Webapp should allow the users to change the maketing inputs 
5. **Code Quality**: Pass `ruff` linting checks, proper error handling, no hardcoded values
6. **Testing**: Ensure your app runs locally before submission,Unit tests for core optimization logic and API endpoints
8. **Documentation**: Updated README.md and mkdocs documentation for your submission
9. **Evaluation branch**  Default branch (main/master) will be considered for the Quality evalution.


## Webapp functional requirements:
[View the detailed functional requirements](./functional_requirements.md)

## 🎯 Evaluation Criteria

Your submissions will be evaluated on:

1. **Local Setup & Execution** (25%) - Can we run your app locally using your README?
2. **Interactive Functionality** (20%) - Table display and dynamic filtering work correctly
3. **Code Quality** (15%) - Passes linting, clean structure, proper error handling
4. **Testing Implementation** (15%) - Unit tests and API tests included
5. **Documentation Quality** (15%) - Clear README and mkdocs documentation


## Documentation updates

### 1. Update your README.md with webapp setup instructions 

Replace the contents from README.md as per [this template](./README_template.md).
Your README.md will be used by the Hackathon platform team for setting up your project for evaluation.

**Focus**: Setup instructions, installation, running the app, and troubleshooting.

### 2. Use MKDOCS for project presentation

**Guide**: See [MKDOCS_GUIDE.md](./MKDOCS_GUIDE.md) for detailed instructions on customizing your documentation site.


Happy coding! 🚀