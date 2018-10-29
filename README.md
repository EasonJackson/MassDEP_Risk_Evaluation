# MassDEP_Risk_Evaluation
GQP_Fall2018

## Workflow
### 1.Text Cleaning
 - Replacing
 
    Items | Example
    ---|---
    Abbreviations | "Immediate Response Action (IRA)", <br>"Massachusetts Department of Environmental Protection (DEP)", <br>"Substantial Release Migration (SRM)"
    Addresses | "218 South Street in Auburn, Massachusetts", <br>"218 South Street, Auburn MA 01501"
    Dates and Time | "June 2017", <br>"June 16, 2017" <br>"9:57 a.m.", <br>"On June 13 & 14, 2017"
    **Numbers** | ***
    Longitudes and Latitudes | "42o10'36" north latitude (42.17650 °N), 71o50'19" east longitude (-71.83856 °W)" 
    Regulation and Forms | "310 CMR 40.0420(7)", <br>"Forms BWSC 123"
    Measurements | "95,832 square feet (approximately 2.20 acres)", <br>"approximately 50' south of the release area", <br>"6-7' below grade", <br>"<10 ppmv", <br>"within 1⁄2-1' of the water table"
    RTN Numbers | "RTN 2-20220"
    Chemicals | 
    Legal Terms Quotes | "‘significant risk’", <br>"“level of diligence reasonably necessary to obtain the quantity and quality of information adequate to assess”"
    Names | "Robert L. Haroian"
    
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
    * Skip-gram selecting input words and targets
 Skip-gram:
 
 ![PH](/Skip_gram.png =300x200 "Skip-gram")
 
 ### 3.Word2Vec Model
 ![PH](/word2vec_model.png =300x200 "Word2Vec")
