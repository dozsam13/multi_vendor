# Data sources

This project is a collaboration with the Heart and Vascular Center of Semmelweis University. 
As a consequence a lot of data come from the Center but in order to examine the effect of the heterogeneous dataset, more additional sources were downloaded.
The currently available sources are listed here and the most important properties are also included.
Such properties are: gender availibility (av.), pathology av., height-weight av., gap between slices, which part of the heart has ground truth contours.

----------------------
Varosmajor
----------------------

**Properties:**
- Gender: YES
- Pathology: YES
- Height-Weight: YES
- Gap between slices: no
- Size: 5400

![parts](images/rnrplnlp.png)


**General information:**

High quality database with 5400 patient data.

----------------------
SunnyBrooks 
----------------------

**Properties:**
- Gender: YES
- Pathology: YES
- Height-Weight: no
- Gap between slices: yes
- Size: 45

![parts](images/lnlp.png)

**General information:**

Source link:
http://www.cardiacatlas.org/studies/sunnybrook-cardiac-data/

Should be cited in the following way:
Radau P, Lu Y, Connelly K, Paul G, Dick AJ, Wright GA. "Evaluation Framework for Algorithms Segmenting Short Axis Cardiac MRI." The MIDAS Journal - Cardiac MR Left Ventricle Segmentation Challenge, http://hdl.handle.net/10380/3070

----------------------
MICCAI2012
----------------------

**Properties:**
- Gender: YES
- Pathology: no
- Height-Weight: YES
- Gap between slices: yes
- Size: 16

![parts](images/rnrp.png)

**General information:**

Source link: 
http://www.litislab.fr/?sub_project=how-to-download-the-data

----------------------
MICCAI2017
----------------------

**Properties:**
- Gender: no
- Pathology: YES
- Height-Weight: YES
- Gap between slices: unknown, assume 0
- Size: 100

![parts](images/rnlnlp.png)

**General information:**

Source link:
https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html

The data was acquired by two MRI scanners! 

----------------------
STACOM2011
----------------------

**Properties:**
- Gender: no
- Pathology: no
- Height-Weight: no
- Gap between slices: yes
- Size: 100

![parts](images/lnlp.png)

**General information:**

Source link:
http://www.cardiacatlas.org/challenges/lv-segmentation-challenge/
http://www.cardiacatlas.org/studies/determine/

----------------------
STACOM2019
----------------------

**Properties:**
- Gender: no
- Pathology: no
- Height-Weight: no
- Gap between slices: no
- Size: 56

![parts](images/lnlp.png)

**General information:**

Source link:
https://lvquan19.github.io/
 
Additional information:
Many participants are confused about the way to transfer quantification values to physical values, here are the equations may help you to finish this task:
For example, areas of patient1 time1, the real physical  value is 0.2368*(1.4062*80)^2=2996.7869mm^2.
 
And the equation of physical value of rwt and dims is：
 
                            physical value of rwt= rwt*pix_spacing*80
                            physical value of dims= dims*pix_spacing*80
 
And the equation of physical value of areas is：
physical value of areas= areas*(pix_spacing*80)^2
 
It should be noted that all the value should multiply 80 or 80^2, not 256 or 512. This is because all the ground truth come from the data provided last year, and all the pictures have  been normalized to 80*80 last year.
