#include "../include/dynamics_models/stapler_dynamics.h"


namespace dynamics
{

/* Stapler in Revolute mode, No Contact */
class StaplerDynamicsC : public StaplerDynamics{
    public:
        StaplerDynamicsC(int n_x_rev, int n_u_rev, int n_z_rev, int n_x_free, int n_u_free, int n_z_free) : StaplerDynamics(n_x_rev, n_u_rev, n_z_rev, n_x_free, n_u_free, n_z_free){};


        // Methods
        void propagateState(Eigen::VectorXd x, Eigen::VectorXd u, Eigen::VectorXd& x_new)
        {
            tf::Transform t = tf::Transform(tf::Quaternion(0.,0.,0.,1.), tf::Vector3(0.,0.,0));
            propagateStateUseTransform(x, u, x_new, t);
        }


        void propagateStateUseTransform(Eigen::VectorXd x, Eigen::VectorXd u, Eigen::VectorXd& x_new, tf::Transform t = tf::Transform(tf::Quaternion(0.,0.,0.,1.), tf::Vector3(0.,0.,0)))
        {
            Eigen::VectorXd x_new_free;

            freebody->propagateState(x.head(3), u, x_new_free);
            x_new.head(3) = x_new_free;

            // Obstacles restrict motion
            if(x[1] >= y_obs)
                x_new[1] = x[1];

            x_new[3] = theta_max*scale;
        }

        void getObservation(Eigen::VectorXd x, Eigen::VectorXd& z)
        {
            Eigen::VectorXd z_free;
            freebody->getObservation(x.head(3), z_free);
            z.head(3)  = z_free;

            z[3] = x[3];
        }

        Eigen::MatrixXd V_mat() {
            return 0.02 * (scale * scale) * Eigen::MatrixXd::Identity(4, 4);
        }

        Eigen::MatrixXd W_mat() {
            return 0.005 * (scale * scale) * Eigen::MatrixXd::Identity(4, 4);
        }
};
}