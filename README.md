# Linear Regression App
<img width="302" height="149" alt="linear_regression_app_readme_image_cover" src="https://github.com/user-attachments/assets/49c57a35-560f-49b8-92e1-a9daa64ded0e" />

This project is a Python-based desktop application that allows users to:

- Load data from **CSV**, **Excel**, or **SQLite** files  (Input data)
- Create and visualize **simple and multiple linear regression models** (Model)  
- Perform **predictions** using trained models  (Prediction)
- Save and load models
- Test the quality of a model with a parity graph  
- Use a **graphical user interface** PyQt5 to execute all actions easily

This software was made using the **Scrum methodology**, along with **Git**, **GitHub**, and **Taiga** to ensure the effective colaboration between the group members.  

### Developed by:
- Álvaro Gómez García
- Sebastián David Moreno Expósito
- Sergio Pérez Vilar
- Xoel Sánchez Dacoba

# How to Download and Run the Project
 
 ### 1. Requirements

  - Python *3.Y*
  - pip
  - Dependencies (install using pip):  
  ``` pip install -r dependencies.txt ```  
    You'll find **dependencies.txt** inside the root folder of the project  
  - (Note): You may encounter problems installing these dependencies, if this is the case please consider
    updating your pip using: ```python -m install --upgrade pip``` and running the previous command again.
    If this doesn't work please consider installing a C/C++ compiler in your PC, then run again the pip command.
    But at today's date is already tested that isn't necessary to do so.

  ### 2. Download and run the program
- Download the .zip file from the release section, once downloaded extract it and run main
- Alternatively: Clone the repo from your desire folder in your PC and run main

# User guide
### **1. Load data**
- Use the button **"Open File"** in Data Management to open a file  
  <img width="570" height="113" alt="1" src="https://github.com/user-attachments/assets/8eb39cab-6946-4742-8861-9150529efa15" />

- Select between **CSV**, **Excel**, and **MySQL** data formats to load
  <img width="766" height="318" alt="2" src="https://github.com/user-attachments/assets/4c0f1d88-6fd6-486d-bdcb-672464debeed" />

---

### **2. Select features and target**
- Select input (**features**) columns  
- Select output (**target**) column  
  <img width="638" height="557" alt="3" src="https://github.com/user-attachments/assets/9529d92f-33b7-4e0d-93f1-8c91a3ad6ec7" />

---

### **3. Confirm your selection**
- Press the **"Confirm Selection"** button  
  <img width="631" height="202" alt="4" src="https://github.com/user-attachments/assets/e4af1e44-56c6-4d17-a999-2b44d915717c" />

---

### **4. If you selected columns with NaN values**
- A new option to the right will appear. Open the dropdown and select one of the four options
   <img width="632" height="199" alt="5" src="https://github.com/user-attachments/assets/48602c8d-c099-4720-82f3-ffed4a452915" />


- Then press **"Apply preprocessing"** to handle your missing data"
  <img width="629" height="200" alt="6" src="https://github.com/user-attachments/assets/65e0996d-4af7-4a71-a1e4-aa7fb539b607" />

---

### **5. Split data into train and test sets**
- Fill both input fields

  <img width="788" height="197" alt="7" src="https://github.com/user-attachments/assets/d3595678-7777-49fc-81a7-e9b225488141" />


- Use the **"Create Model"** button to split your data  
  and create your linear regression model
  <img width="788" height="197" alt="8" src="https://github.com/user-attachments/assets/c5ae4e60-acf7-4751-988f-fb24420f80c6" />

- An informative pop-up message will apper once you press **"Create Model"** button

  <img width="227" height="181" alt="9" src="https://github.com/user-attachments/assets/2507fbeb-6c33-4daf-a31a-ca10681a4a29" />

- A summary message will appear next to where  
  you pressed **"Create Model"**, containing a summary of your model  
  <img width="764" height="183" alt="10" src="https://github.com/user-attachments/assets/34c7035a-ebf1-4ef6-a5fe-10a551a59de5" />

---
### **6. See your linear regression**
- Hit the **"Model Management"** button on the top bar
  <img width="789" height="75" alt="11" src="https://github.com/user-attachments/assets/9d995ef6-e2ed-47c9-ab6c-2a878bc7230b" />

- There you'll find your **parity graph** and **simple regression graph**  
  (only simple regression shows the second graph)  
- Consult useful metrics such as **R²** and **MSE** for both train and test
  <img width="1095" height="609" alt="12" src="https://github.com/user-attachments/assets/c4b04e3b-1fe4-4ae8-9f01-0e0f89b634e7" />


---

### **7. Make a prediction**
- Use the **"Make a Prediction"** section by assigning values  
  to the input values and pressing **Make prediction** button
  this will show you the predicted value for the given input values

  <img width="627" height="177" alt="13" src="https://github.com/user-attachments/assets/eb3631ec-e289-42bc-95b7-2404abbd16a0" />


---

### **8. Save your model**
- You may add a description before saving your model  
- To save it, press the **"Save Model"** button and set a directory to save it (you save the model, not the graph/s)
  <img width="1083" height="595" alt="14" src="https://github.com/user-attachments/assets/bec72fdf-675a-47ba-a6c5-a5871464b304" />

---

### **9. Load your model**
- Hit the **"Load Model"** button and select a previously saved model
- This will load the model on Model Management section.

### Thanks for reading we hope that this guide was helpful. Enjoy the program.
