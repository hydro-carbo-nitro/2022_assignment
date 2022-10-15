# LinearFitting

> Reference : Prof. Yook Lecture.
> [http://fracton.khu.ac.kr/~syook/Lectures/ComputPhys/Fitting.pdf]

## Theory

Consider $N$samples. Then, we have $N$ of set $\{x^{(i)}, y^{(i)}, \sigma^{(i)} | i \leq N\}$.
Then, if $y^{(i)}$ is able to express with a function of $x^{(i)}$ with $m+1$ parameters,  we want to find the best parameter set which can express $y$. 
$$
\begin{aligned}
	y &= f(A; x)\qquad A=\{a_0, a_1, \cdots, a_m\} 
\end{aligned}
$$
The reason why the number of parameters is $m+1$, not $m$, is that I will fit the data on $m$ order linear equation. (i.e. $f(x) = \sum_{n=0} ^{m} a_nx^n$)

Now, we discriminate $y$ and $f$. $y$ is the **measured value** and $f$ is **predicted value**. Measured value $y^{(i)}$ has an inevitable error $\sigma^{(i)}$by the **central limit theorem**
$$
\begin{aligned}
	p(y_i)dy_i	&= \frac{1}{\sigma_i\sqrt{2\pi}}\exp\{{(y_i - f_i)^2 / 2\sigma_i^2}\} dy_i \\
	f_i				&= f(A;x_i)
\end{aligned}
$$


Given situation, the likelihood function for $\{y_i\}$ is defined as
$$
\begin{aligned}
	\mathcal{L}	&=	\prod_{i=1}^N p(y_i) \\
							&=	\prod_{i=1}^N 
									\frac{1}{\sigma_i\sqrt{2\pi}}
									\exp\{{(y_i - f_i)^2 / 2\sigma_i^2}\} \\
							&=	\left\{\prod_{i=1}^N 
									\frac{1}{\sigma_i\sqrt{2\pi}}\right\}
									\exp\left[ -{{1}\over{2}}
									\sum_{i=1}^N
									{\left( \frac{y_i - f_i}
									{\sigma_i}\right)}^2\right]\\
							&=	\left\{\prod_{i=1}^N 
									\frac{1}{\sigma_i\sqrt{2\pi}}\right\}
									\exp \chi^2\\
	\chi^2 			&\equiv 
									\sum_{i=1}^N
									{\left( \frac{y_i - f_i}
									{\sigma_i}\right)}^2
 \end{aligned}
$$

What I want to do is minimizing $\chi^2$ in the parameter space. That is
$$
\begin{aligned}
	\frac{\partial\chi^2}{\partial a_k}
		&=	-2\sum_{i=1}^N
				{\left( \frac{y_i - f_i}{\sigma_i^2}\right)}
				\frac{\partial f_i}{\partial a_k} = 0
		\tag1 
\end{aligned}
$$  
where $k = 0, 1, 2,\cdots ,m$

From the form of $f_i$ and Eq. (1), we can get this equation.
$$
\begin{aligned}
	&\frac{\partial f}{\partial a_k} =x^k\\
	&\sum_{i=1}^N
				{\left( \frac{y_i - f_i}{\sigma_i^2}\right)}x_i^k = 0
				\tag2
\end{aligned}
$$ 

Think about case of $m=2$. From Eq.(2), we can get 3 equations.
$$
\begin{aligned}
	&\sum_{i=1}^N
		{\left( \frac{y_i - f_i}{\sigma_i^2}\right)} &= 0\\
	&\sum_{i=1}^N
		{\left( \frac{y_i - f_i}{\sigma_i^2}\right)}x_i &= 0\\
	&\sum_{i=1}^N
		{\left( \frac{y_i - f_i}{\sigma_i^2}\right)}x_i^2 &= 0
		\tag 3
\end{aligned}
$$
Now, we have linear equation of $f_i$ and from Eq. (3), we can get these equations.
$$
\begin{aligned}
	f_i	&= \sum_{k=0}^m a_kx_i^k \\
			&= a_0 + a_1x+a_2x^2
\end{aligned}
$$
$$
\begin{aligned}
	&a_0\sum_{i=1}^N{\frac{1}{\sigma_i^2}}
	+a_1\sum_{i=1}^N{\frac{x_i}{\sigma_i^2}}
	+a_2\sum_{i=1}^N{\frac{x_i^2}{\sigma_i^2}}
	=\sum_{i=1}^N{\frac{y_i}{\sigma_i^2}}\\
	&a_0\sum_{i=1}^N{\frac{x_i}{\sigma_i^2}}
	+a_1\sum_{i=1}^N{\frac{x_i^2}{\sigma_i^2}}
	+a_2\sum_{i=1}^N{\frac{x_i^3}{\sigma_i^2}}
	=\sum_{i=1}^N{\frac{x_iy_i}{\sigma_i^2}}\\
	&a_0\sum_{i=1}^N{\frac{x_i^2}{\sigma_i^2}}
	+a_1\sum_{i=1}^N{\frac{x_i^3}{\sigma_i^2}}
	+a_2\sum_{i=1}^N{\frac{x_i^4}{\sigma_i^2}}
	=\sum_{i=1}^N{\frac{x_i^2y_i}{\sigma_i^2}}\\
\end{aligned}
$$
we can express this equations as matrix form
*this expression is on myself*
$$
\begin{aligned}
	&\begin{pmatrix}
		\theta_0 & \theta_1  & \theta_2 \\
		\theta_1 & \theta_2  & \theta_3 \\
		\theta_2 & \theta_3  & \theta_4 \\
	\end{pmatrix}
	\begin{pmatrix}
		a_0\\
		a_1\\
		a_2\\
	\end{pmatrix}
	=
	\begin{pmatrix}
		\phi_0\\
		\phi_1\\
		\phi_2\\
	\end{pmatrix}\\
	&
	\theta_k =	\sum_{i=1}^N \frac{x_i^k}{\sigma_i^2}, \quad
	\phi_k		=	\sum_{i=1}^N \frac{x_i^ky_i}{\sigma_i^2}
\end{aligned}
$$
I can express this as 
$$
\begin{aligned}
	&\Theta\vec{a}=\vec{\phi} \\
	&\Theta = \mathbf{Mat}(m+1;R), \quad\vec{a}, \vec{\phi} \in R^{m+1}
\end{aligned}
$$
Suprisingly $\Theta$ is symmetric matrix. Is there something special? I don't know. Maybe not. Anyway, we have a data about $\Theta$ and $\vec{\phi}$ . What we want to know is $\vec{a}$. To solve this problem, we apply **GaussElimination**.

## Algorithm
**1. Bring data**
In my code, it reads 3 columns data $\{x_i, y_i, \sigma_i\}$ and put it in **LinearSquareFit**

**2. Fitting**
I want more efficient and vectorized code but I'm used to write C code, so it is so hard to me. Anyway, the target is caculate $\Theta$ and $\vec{\phi}$ with dataset. The way how to caculate them is written above thoery part.

**3. GaussElimination**
As I wrote, what I need to do is solve $\vec{a}$ from $\Theta \vec{a}=\vec{\phi}$. Before applying gauss elimination, if $\Theta$'s $(1, 1)$ element is zero, I think this situation is never happened, we cannot apply gauss elimination. To prevent this situation, we apply partial pivoting which changes the row of elements to prevent be divided by zero.

**4.Plot**
AND DRAW! PRETTY!
