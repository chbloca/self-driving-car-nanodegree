#include <iostream>

// class declaration
class Gaussian
{
    private:
        float mu, sigma2;

    public:

        // constructor functions
        Gaussian ();
        Gaussian (float, float);

        // changes value of average and standard deviation
        void setMu(float);
        void setSigma2(float);

        // outputs value of average and standard deviation
        float getMu();
        float getSigma2();

        // class methods
        float evaluate (float);
        Gaussian mul (Gaussian);
        Gaussian add (Gaussian);
};


int main ()
{

 /*
TODO: Instantiate a Gaussian object called gaussianone.The object
should have mean = 40.0 and variance (aka sigma2) = 225.0
*/

Gaussian gaussianone(40.0, 225.0);

/*
TODO: Instantiate another Gaussian object called gaussiantwo.
The object should have mean = 35.6 and variance = 12.25
*/

Gaussian gaussiantwo(35.6, 12.25);

/*
TODO:
Output to the terminal the following (hint: use the std namespace with cout or
use std::cout):
- the probability density function value for gaussianone when x = 10.5
- the probability density function value for gaussianone when x = 55.4
- the probability density function value for gaussiantwo when x = 35.6
- the probability density function value for gaussiantwo when x = 29.4
*/

std::cout << "the pdf value for gaussianone when x = 10.5 is: " << gaussianone.evaluate(10.5) << std::endl;
std::cout << "the pdf value for gaussianone when x = 55.4 is: " << gaussianone.evaluate(55.4) << std::endl;
std::cout << "the pdf value for gaussianone when x = 35.6 is: " << gaussianone.evaluate(35.6) << std::endl;
std::cout << "the pdf value for gaussianone when x = 29.4 is: " << gaussianone.evaluate(29.4) << std::endl;

/*
TODO:
- Change the mean value of gaussianone to mean = 45
- Change the variance of gaussiantwo to variance = 15.4
- Output the mean of gaussianone to the terminal
- Output the variance of gaussiantwo to the terminal
*/
gaussianone.setMu(45);
gaussiantwo.setSigma2(15.4);
std::cout << gaussianone.getMu() << std::endl;
std::cout << gaussiantwo.getSigma2() << std::endl;

/*
TODO:
- Multiply gaussian one and gaussian two. Store the resulting gaussian
in a variable called gaussianthree
- Output the mean and variance of gaussianthree to the terminal
- Add gaussian one and gaussian two. Store the resulting gaussian in a
variable called gaussianfour
- Output the mean and variance of gaussianfour to the terminal
*/

Gaussian gaussianthree = gaussianone.mul(gaussiantwo);

std::cout << "gaussianthree has mean: " << gaussianthree.getMu() << " and variance: " << gaussianthree.getSigma2() << std::endl;

Gaussian gaussianfour = gaussianone.add(gaussiantwo);

std::cout << "gaussianthree has mean: " << gaussianfour.getMu() <<" and variance: " << gaussianfour.getSigma2() << std::endl;


}
