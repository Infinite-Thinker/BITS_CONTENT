Work Integrated Learning Programmes Division M. Tech. in Artificial Intelligence & Machine Learning I Semester 2026-27 

# **Assignment I** 

## **AIMLZC416 — Mathematical Foundations for Machine Learning** 

## **Instructions** 

1. Read through the FAQ document posted on the LMS to go through the points related to assignments and submissions. 

2. Use only Python programming language wherever necessary and attach the written code in your submission. These can be screenshot of the code or the code pasted in a document or handwritten. 

3. Numerical output can also be screenshots or it can be pasted on to document or handwritten. 

4. Python code without comment lines giving the description would not be awarded marks. 

5. Use of built-in functions is not allowed unless it is specifically stated. 

6. All entities in the assignment should be random entries of the form _n.dddddddd_ , eg 3.47681539. No marks would be awarded for random integer entries. 

7. All numerical entities used need to be shown in your submissions. 

8. This is not a group activity. Each student should do the problems and submit individually. 

9. Assignments should be uploaded as a single pdf (and only as pdf) with name BITSID.pdf. 

10. The size restriction is 10 MB per submission. 

11. Submission is allowed only once. Ensure that the correct file is uploaded. 

12. Submissions beyond 6:59 pm on 30-08-2026 would not be graded. 

13. Do not wait till the last moment to upload. Try to finish the upload by 6:30 pm itself. 

14. Assignments sent via email would not be accepted. 

15. Copying is strictly prohibited. Adoption of unfair means would lead to disciplinary action. 

## **Q1) F** inding solutions of linear systems 

1. Write a code taking as input a matrix **_A_** of size _m × n_ and a vector **_b_** of size _m ×_ 1, where _m_ and _n_ are arbitrarily large numbers and _m < n_ , constructing the augmented matrix and performing 

   -  REF, and 

   -  RREF 

without using any built-in functions. In case you encounter any division by 0, you can choose a different **_A_** and/or **_b_** . Note that this part should be explicitly there in the code and give a comment line on the same. (1 mark + 1 mark) 

2. Write a Python code to identify the pivot and non-pivot columns and find the particular solution and solutions to **_Ax_** = **0** . (1 mark) 

3. Consider a random 5 _×_ 7 matrix **_A_** and a suitable **_b_** and show the REF, RREF, pivot columns, non-pivot columns, the particular solution, the solutions to **_Ax_** = **0** , the general solution and verify the general solution. (1/4 _×_ 8 = 2 marks) 

**Q2)** Consider a dataset _X ∈_ R<sup>500</sup><sup>_×_6</sup> constructed as follows: the first four features _f_ 1 _, f_ 2 _, f_ 3 _, f_ 4 are generated as random features sampled from a standard normal distribution (read about this). The fifth and sixth features are defined by the relations 



Perform the following tasks in sequence: 

1. Write a Python code to generate the dataset _X_ = [ _f_ 1 _, f_ 2 _, f_ 3 _, f_ 4 _, f_ 5 _, f_ 6]. [0.5] 

2. Write a Python code which computes the rank of _X_ and display the output for the dataset _X_ generated in step 1. [0.5] 

3. **Numerical Experiment with the Power Method** 

   - Read about the power method for finding the dominant eigenvalue and its corresponding eigenvector and perform the following tasks. 

   - (a) Write a Python code to compute the covariance matrix [0.5] 



where _n_ is number of data points 

- (b) Implement the Power Method in Python to approximate the largest eigenvalue _λ_ 1 and its corresponding eigenvector _v_ 1 of _C_ . Show the code and the outputs. [1] 

- (c) Write a Python code to obtain the next largest eigenvalue _λ_ 2 and its corresponding eigenvector _v_ 2 by applying power method on _C − v_ 1 _v_ 1<sup>_TC_.</sup> Having found out _v_ 1 _, v_ 2 _, . . . , vk−_ 1, one can find _λk_ and its corresponding eigenvector _vk_ by applying power method on _C −_<sup>�</sup><sup>_k_</sup> _j_ =1<sup>_−_1</sup><sup>_vjv_</sup> _J_<sup>_TC_.Givethecodeandalsodisplaytheobtained</sup> eigenvalues and the corresponding eigenvectors. [1.5] 

- (d) Find all the eigenvalues and eigenvectors using Python function and compare with the obtained result in (c). [0.5] 

- (e) Compare the number of iterations required to get an accuracy of 10<sup>_−_7</sup> using the power method. The actual values can be taken as the one obtained in (d). [0.5] 

2 

