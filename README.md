# MassDEP_Risk_Evaluation
GQP_Fall2018

## Workflow
### 1.Text Cleaning
 - Replacing
 
    Items | Example
    ---|---
    Abbreviations | "Immediate Response Action (IRA)", <br>"Massachusetts Department of Environmental Protection (DEP)", <br>"Substantial Release Migration (SRM)"
    Addresses | "218 South Street in Auburn, Massachusetts", <br>"218 South Street, Auburn MA 01501"
    Dates | "June 2017"
    Numbers |
    No. of street | 
    Longitude & latitude | "42o10'36" north latitude (42.17650 °N), 71o50'19" east longitude (-71.83856 °W)"
    Inches and feet |
    Regulation numbers |
    Measurements | 
    Regulations
    Chemicals
    Legal terms quotes
    
 - Removing
    * Tables and forms within text part
    * Footnotes
    * Captions within text part
    * Special characters ("&", and so on)

### 2.Data Preparation
 - Compose data with list of sentences
 - Compose sentence with list of words
 - Mini batch GD
    * Randomly select sentences
    * Linearly select input words from sentence
    * 
