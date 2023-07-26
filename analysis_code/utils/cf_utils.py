import numpy as np
import cortex
from scipy import stats
from scipy.stats import pearsonr, zscore

def stimulus_through_prf(prfs, stimulus, dx, mask=None):
    """stimulus_through_prf
    dot the stimulus and the prfs
    Parameters
    ----------
    prfs : numpy.ndarray
        the array of prfs. 
    stimulus : numpy.ndarray
        the stimulus design matrix, either convolved with hrf or not.
    mask : numpy.ndarray
        a mask in feature space, of dimensions equal to 
        the spatial dimensions of both stimulus and receptive field
    """
    assert prfs.shape[1:] == stimulus.shape[:-1], \
        """prf array dimensions {prfdim} and input stimulus array dimensions {stimdim} 
        must have same dimensions""".format(
            prfdim=prfs.shape[1:],
            stimdim=stimulus.shape[:-1])
    if mask == None:
        prf_r = prfs.reshape((prfs.shape[0], -1))
        stim_r = stimulus.reshape((-1, stimulus.shape[-1]))
    else:
        assert prfs.shape[1:] == mask.shape and mask.shape == stimulus.shape[:-1], \
            """mask dimensions {maskdim}, prf array dimensions {prfdim}, 
            and input stimulus array dimensions {stimdim} 
            must have same dimensions""".format(
                maskdim=mask.shape,
                prfdim=prfs.shape[1:],
                stimdim=stimulus.shape[:-1])
        prf_r = prfs[:, mask]
        stim_r = stimulus[mask, :]
    return prf_r @ stim_r * (dx ** len(stimulus.shape[:-1]))

class subsurface(object):
    
    """subsurface
        This is a utility that uses pycortex for making sub-surfaces for CF fitting.
        
    """

    def __init__(self,cx_sub,boolmasks,surftype='fiducial'):
        
        """__init__
        Parameters
        ----------
        cx_sub : The name of the cx subject (string). This is used to get surfaces from the pycx database.
        boolmasks: A list of boolean arrays that define the vertices that correspond to the ROI one wants to make a subsurface from [left hem, right hem].
        surftype: The surface (default = fiducial).
        """
        
        self.cx_sub=cx_sub
        self.surftype=surftype
        self.boolmasks=boolmasks
        self.mask=np.concatenate([self.boolmasks[0],self.boolmasks[1]]).astype(int) # Put the mask into int format for plotting.
        

        
    def create(self):
        """get_surfaces
        Function that creates the subsurfaces.
        """
        
        self.get_surfaces()
        self.generate()
        self.get_geometry()
        self.pad_distance_matrices()
        
    def get_surfaces(self):
        
        """get_surfaces
        Accesses the pycortex database to return the subject surfaces (left and right).
        
        Returns
        -------
        subsurface_L, subsurface_R: A pycortex subsurfaces classes for each hemisphere (These are later deleted by 'get_geometry', but can be re-created with a call to this function).
        self.subsurface_verts_L,self.subsurface_verts_R : The whole brain indices of each vertex in the subsurfaces.
        
        """
        
        self.surfaces = [cortex.polyutils.Surface(*d)
        for d in cortex.db.get_surf(self.cx_sub, self.surftype)]
                
    def generate(self):
        
        """generate
        Use the masks defined in boolmasks to define subsurfaces.
        """
        
        print('Generating subsurfaces')
        self.subsurface_L = self.surfaces[0].create_subsurface(vertex_mask=self.boolmasks[0]) # Create sub-surface, left hem.
        self.subsurface_R = self.surfaces[1].create_subsurface(vertex_mask=self.boolmasks[1]) # Create sub-surface, right hem.
        
        # Get the whole-brain indices for those vertices contained in the subsurface.
        self.subsurface_verts_L=np.where(self.subsurface_L.subsurface_vertex_map!=stats.mode(self.subsurface_L.subsurface_vertex_map)[0][0])[0]
        self.subsurface_verts_R=np.where(self.subsurface_R.subsurface_vertex_map!=stats.mode(self.subsurface_R.subsurface_vertex_map)[0][0])[0]+self.subsurface_L.subsurface_vertex_map.shape[-1]
        

    def get_geometry(self):
        
        """get_geometry
        Calculates geometric info about the sub-surfaces. Computes geodesic distances from each point of the sub-surface.
        
        Returns
        -------
        dists_L, dists_R: Matrices of size n vertices x n vertices that describes the distances between all vertices in each hemisphere of the subsurface.
        subsurface_verts: The whole brain indices of each vertex in the subsurface.
        leftlim: The index that indicates the boundary between the left and right hemisphere. 
        """
        
        # Assign some variables to determine where the boundary between the hemispheres is. 
        self.leftlim=np.max(self.subsurface_verts_L)        
        self.subsurface_verts=np.concatenate([self.subsurface_verts_L,self.subsurface_verts_R])
        
        # Make the distance x distance matrix.
        ldists,rdists=[],[]

        print('Creating distance by distance matrices')
        
        for i in range(len(self.subsurface_verts_L)):
            ldists.append(self.subsurface_L.geodesic_distance([i]))
        self.dists_L=np.array(ldists)
        
        for i in range(len(self.subsurface_verts_R)):
            rdists.append(self.subsurface_R.geodesic_distance([i]))
        self.dists_R=np.array(rdists)
        
        self.surfaces,self.subsurface_L,self.subsurface_R=None,None,None # Get rid of these as they are harmful for pickling. We no longer need them.
        
        
    def pad_distance_matrices(self,padval=np.Inf):
        
        """pad_distance_matrices
        Pads the distance matrices so that distances to the opposite hemisphere are np.inf
        Stack them on top of each other so they will have the same size as the design matrix
       
        
        Returns
        -------
        distance_matrix: A matrix of size n vertices x n vertices that describes the distances between all vertices in the subsurface.
        """
        
        padL=np.pad(self.dists_L, ((0, 0), (0, self.dists_R.shape[-1])),constant_values=np.Inf) # Pad the right hem with np.inf.
        padR=np.pad(self.dists_R, ((0, 0), (self.dists_L.shape[-1],0)),constant_values=np.Inf) # pad the left hem with np.inf..
        
        self.distance_matrix=np.vstack([padL,padR]) # Now stack.
        
    def elaborate(self):
        
        """elaborate
        Prints information about the created subsurfaces.
        """
            
        print(f"Maximum distance across left subsurface: {np.max(self.dists_L)} mm")
        rint(f"Maximum distance across right subsurface: {np.max(self.dists_R)} mm")
        print(f"Vertices in left hemisphere: {self.dists_L.shape[-1]}")
        print(f"Vertices in right hemisphere: {self.dists_R.shape[-1]}")

        
class CFStimulus(object):
    
    """CFStimulus
    Minimal CF stimulus class. Creates a design matrix for CF models by taking the data within a sub-surface (e.g. V1).
    """
    
    def __init__(self,
                 data,
                 vertinds,
                 distances,**kwargs):
        
        """__init__
        Parameters
        ----------
        data : numpy.ndarray
            a 2D matrix that contains the whole brain data (second dimension must be time). 
            
        vertinds : numpy.ndarray
            a matrix of integers that define the whole-brain indices of the vertices in the source subsurface.
            
        distances : numpy.ndarray
            a matrix that contains the distances between each vertex in the source sub-surface.
            
        Returns
        -------
        data: Inherits data.
        subsurface_verts: Inherits vertinds.
        distance_matrix: Inherits distances.
        design_matrix: The data contained within the source subsurface (to be used as a design matrix).
        
        """
        self.data = data
        self.subsurface_verts = vertinds
        self.design_matrix = self.data[self.subsurface_verts,:]
        self.distance_matrix = distances


class CFGaussianModel():
    
    """A class for constructing gaussian connective field models.
    """
    
    def __init__(self,stimulus):
            
        """__init__
        
        Parameters
        ----------
        stimulus: A CFstimulus object.
        """
        
        self.stimulus=stimulus
        
    def create_rfs(self):
        
        """create_rfs
        creates rfs for the grid search
        
        Returns
        ----------
        grid_rfs: The receptive field profiles for the grid. 
        vert_centres_flat: A vector that defines the vertex centre associated with each rf profile.
        sigmas_flat: A vector that defines the CF size associated with each rf profile.
        
        
        """
        
        assert hasattr(self, 'sigmas'), "please define a grid of CF sizes first."
        
        if self.func=='cart':
            
            # Make the receptive fields extend over the distances controlled by each of the sigma.
            self.grid_rfs  = np.array([gauss1D_cart(self.stimulus.distance_matrix, 0, s) for s in self.sigmas])
        
        # Reshape.
        self.grid_rfs=self.grid_rfs.reshape(-1, self.grid_rfs.shape[-1])
        
        # Flatten out the variables that define the centres and the sigmas.
        self.vert_centres_flat=np.tile(self.stimulus.subsurface_verts,self.sigmas.shape)
        self.sigmas_flat=np.repeat(self.sigmas,self.stimulus.distance_matrix.shape[0])
        
        
    def stimulus_times_prfs(self):
        """stimulus_times_prfs
        creates timecourses for each of the rfs in self.grid_rfs
        
         Returns
        ----------
        predictions: The predicted timecourse that is the dot product of the data in the source subsurface and each rf profile.
        
        """
        
        assert hasattr(self, 'grid_rfs'), "please create the rfs first"
        self.predictions = stimulus_through_prf(
            self.grid_rfs, self.stimulus.design_matrix,
            1)
        
    def create_grid_predictions(self,sigmas,func='cart'):
        
        """Creates the grid rfs and rf predictions
        """
        
        self.sigmas=sigmas
        self.func=func
        self.create_rfs()
        self.stimulus_times_prfs()
        
    def return_prediction(self,sigma,beta,baseline,vert):
        
        """return_prediction
        Creates a prediction given a sigma, beta, baseline and vertex centre.
        
        Returns
        ----------
        
        A prediction for this parameter combination. 
        
        """
        
        beta=np.array(beta)
        baseline=np.array(baseline)
        
        # Find the row of the distance matrix that corresponds to that vertex.
        idx=np.where(self.stimulus.subsurface_verts==vert)[0][0]
            
        # We can grab the row of the distance matrix corresponding to this vertex and make the rf.
        rf=np.array([gauss1D_cart(self.stimulus.distance_matrix[idx], 0, sigma)])
            
        # Dot with the data to make the predictions. 
        neural_tc = stimulus_through_prf(rf, self.stimulus.design_matrix, 1)
    

        return baseline[..., np.newaxis] + beta[..., np.newaxis] * neural_tc

def gauss1D_cart(x, mu=0.0, sigma=1.0):
    """gauss1D_cart
    gauss1D_cart takes a 1D array x, a mean and standard deviation,
    and produces a gaussian with given parameters, with a peak of height 1.
    Parameters
    ----------
    x : numpy.ndarray (1D)
        space on which to calculate the gauss
    mu : float, optional
        mean/mode of gaussian (the default is 0.0)
    sigma : float, optional
        standard deviation of gaussian (the default is 1.0)
    Returns
    -------
    numpy.ndarray
        gaussian values at x
    """

    return np.exp(-((x-mu)**2)/(2*sigma**2)).astype('float32')



class Fitter:
    """Fitter
    Superclass for classes that implement the different fitting methods,
    for a given model. It contains 2D-data and leverages a Model object.
    data should be two-dimensional so that all bookkeeping with regard to voxels,
    electrodes, etc is done by the user. Generally, a Fitter class should implement
    both a `grid_fit` and an `iterative_fit` method to be run in sequence.
    """

    def __init__(self, data, model, n_jobs=1, **kwargs):
        """__init__ sets up data and model
        Parameters
        ----------
        data : numpy.ndarray, 2D
            input data. First dimension units, Second dimension time
        model : prfpy.Model
            Model object that provides the grid and iterative search
            predictions.
        n_jobs : int, optional
            number of jobs to use in parallelization (iterative search), by default 1
        """
        assert len(data.shape) == 2, \
            "input data should be two-dimensional, with first dimension units and second dimension time"     

            
        self.data = data.astype('float32')
        
        self.model = model
        self.n_jobs = n_jobs

        self.__dict__.update(kwargs)

        self.n_units = self.data.shape[0]
        self.n_timepoints = self.data.shape[-1]

        self.data_var = self.data.var(axis=-1)

    def iterative_fit(self,
                      rsq_threshold,
                      verbose=False,
                      starting_params=None,
                      bounds=None,
                      args={},
                      constraints=None,
                      xtol=1e-4,
                      ftol=1e-4):
        """
        Generic function for iterative fitting. Does not need to be
        redefined for new models. It is sufficient to define
        `insert_new_model_params` or `grid_fit` in the new model Fitter class,
        or provide explicit `starting_params`
        (see Extend_Iso2DGaussianFitter for examples).
        Parameters
        ----------
        rsq_threshold : float
            Rsq threshold for iterative fitting. Must be between 0 and 1.
        verbose : boolean, optional
            Whether to print output. The default is False.
        starting_params : ndarray, optional
            Explicit start for iterative fit. The default is None.
        bounds : list of tuples, optional
            Bounds for parameter minimization. The default is None.
            if bounds are None, will use Powell optimizer
            if bounds are not None, will use LBFGSB or trust-constr
        args : dictionary, optional
            Further arguments passed to iterative_search. The default is {}.
        constraints: list of scipy.optimize.LinearConstraints and/or
            scipy.optimize.NonLinearConstraints
            if constraints are not None, will use trust-constr optimizer
        xtol : float, optional
            if allowed by optimizer, parameter tolerance for termination of fitting
        ftol : float, optional
            if allowed by optimizer, objective function tolerance for termination of fitting
        Returns
        -------
        None.
        """

        self.bounds = np.array(bounds)
        self.constraints = constraints

        assert rsq_threshold>0, 'rsq_threshold must be >0!'

        if starting_params is None:
            assert hasattr(
                self, 'gridsearch_params'), 'First use self.grid_fit,\
            or provide explicit starting parameters!'

            self.starting_params = self.gridsearch_params

        else:
            self.starting_params = starting_params
        
        #allows unit-wise bounds. this can be used to keep certain parameters fixed to a predetermined unit-specific value, while fitting others.
        if len(self.bounds.shape) == 2:
            self.bounds = np.repeat(self.bounds[np.newaxis,...], self.starting_params.shape[0], axis=0)
            
        if not hasattr(self,'rsq_mask'):
            #use the grid or explicitly provided params to select voxels to fit
            self.rsq_mask = self.starting_params[:, -1] > rsq_threshold

        self.iterative_search_params = np.zeros_like(self.starting_params)

        if self.rsq_mask.sum()>0:
            if np.any(self.bounds) != None:
                iterative_search_params = Parallel(self.n_jobs, verbose=verbose)(
                    delayed(iterative_search)(self.model,
                                              data,
                                              start_params,
                                              args=args,
                                              xtol=xtol,
                                              ftol=ftol,
                                              verbose=verbose,
                                              bounds=curr_bounds,
                                              constraints=self.constraints)
                    for (data, start_params, curr_bounds) in zip(self.data[self.rsq_mask], self.starting_params[self.rsq_mask, :-1], self.bounds[self.rsq_mask]))
            else:
                iterative_search_params = Parallel(self.n_jobs, verbose=verbose)(
                    delayed(iterative_search)(self.model,
                                              data,
                                              start_params,
                                              args=args,
                                              xtol=xtol,
                                              ftol=ftol,
                                              verbose=verbose,
                                              bounds=None,
                                              constraints=self.constraints)
                    for (data, start_params) in zip(self.data[self.rsq_mask], self.starting_params[self.rsq_mask, :-1]))            
            
            self.iterative_search_params[self.rsq_mask] = np.array(
                iterative_search_params)
            
                
    def crossvalidate_fit(self,
                          test_data,
                          test_stimulus=None,
                          single_hrf=True):
        """
        Simple function to crossvalidate results of previous iterative fitting.
       
        Parameters
        ----------
        test_data : ndarray
            Test data for crossvalidation.
        test_stimulus : PRFStimulus, optional
            PRF stimulus for test. If same as train data, not needed.
        single_hrf : Bool
            If True, uses the average-across-units HRF params in crossvalidation
        Returns
        -------
        None.
        """

        assert hasattr(
                self, 'iterative_search_params'), 'First use self.iterative_fit,'      
        
        #to hande cases where test_data and fit_data have different stimuli
        if test_stimulus is not None:                
            fit_stimulus = deepcopy(self.model.stimulus)   
            
            self.model.stimulus = test_stimulus            
            self.model.filter_params['task_lengths'] = test_stimulus.task_lengths
            self.model.filter_params['task_names'] = test_stimulus.task_names
            self.model.filter_params['late_iso_dict'] = test_stimulus.late_iso_dict
            
        if self.rsq_mask.sum()>0:
            if single_hrf:
                median_hrf_params = np.median(self.iterative_search_params[self.rsq_mask,-3:-1],
                                               axis=0)
                
                self.iterative_search_params[self.rsq_mask,-3:-1] = median_hrf_params
                
                
            test_predictions = self.model.return_prediction(*list(self.iterative_search_params[self.rsq_mask,:-1].T))
            
            if test_stimulus is not None:
                self.model.stimulus = fit_stimulus
                self.model.filter_params['task_lengths'] = fit_stimulus.task_lengths
                self.model.filter_params['task_names'] = fit_stimulus.task_names
                self.model.filter_params['late_iso_dict'] = fit_stimulus.late_iso_dict  
                
            #calculate CV-rsq        
            CV_rsq = np.nan_to_num(1-np.sum((test_data[self.rsq_mask]-test_predictions)**2, axis=-1)/(test_data.shape[-1]*test_data[self.rsq_mask].var(-1)))

            self.iterative_search_params[self.rsq_mask,-1] = CV_rsq
        else:
            print("No voxels/vertices above Rsq threshold were found.")


        if self.data.shape == test_data.shape:
              
            self.noise_ceiling = np.zeros(self.n_units)
            
            n_c = 1-np.sum((test_data[self.rsq_mask]-self.data[self.rsq_mask])**2, axis=-1)/(test_data.shape[-1]*test_data[self.rsq_mask].var(-1))
            
            self.noise_ceiling[self.rsq_mask] = n_c

class CFFitter(Fitter):
    
    """CFFitter
    Class that implements the different fitting methods
    on a gaussian CF model,
    leveraging a Model object.
    """
    
    
    def grid_fit(self,sigma_grid,verbose=False,n_batches=1000):
        
        
        """grid_fit
        performs grid fit using provided grids and predictor definitions
        Parameters
        ----------
        sigma_grid : 1D ndarray
            to be filled in by user
        verbose : boolean, optional
            print output. The default is False.
        n_batches : int, optional
            The grid fit is performed in parallel over n_batches of units.
            Batch parallelization is faster than single-unit
            parallelization and of sequential computing.
        Returns
        -------
        gridsearch_params: An array containing the gridsearch parameters.
        vertex_centres: An array containing the vertex centres.
        vertex_centres_dict: A dictionary containing the vertex centres.
        """
        
        
        self.model.create_grid_predictions(sigma_grid)
        self.model.predictions = self.model.predictions.astype('float32')
        
        def rsq_betas_for_batch(data, vox_num, predictions,
                                n_timepoints, data_var,
                                sum_preds, square_norm_preds):
            result = np.zeros((data.shape[0], 4), dtype='float32')
            for vox_data, num, idx in zip(
                data, vox_num, np.arange(
                    data.shape[0])):
                # bookkeeping
                sumd = np.sum(vox_data)

                # best slopes and baselines for voxel for predictions
                slopes = (n_timepoints * np.dot(vox_data, predictions.T) - sumd *
                          sum_preds) / (n_timepoints * square_norm_preds - sum_preds**2)
                baselines = (sumd - slopes * sum_preds) / n_timepoints

                # resid and rsq
                resid = np.linalg.norm((vox_data -
                                        slopes[..., np.newaxis] *
                                        predictions -
                                        baselines[..., np.newaxis]), axis=-
                                       1, ord=2)

                #to enforce, if possible, positive prf amplitude
                #if pos_prfs_only:
                #    if np.any(slopes>0):
                #        resid[slopes<=0] = +np.inf

                best_pred_voxel = np.nanargmin(resid)

                rsq = 1 - resid[best_pred_voxel]**2 / \
                    (n_timepoints * data_var[num])

                result[idx, :] = best_pred_voxel, rsq, baselines[best_pred_voxel], slopes[best_pred_voxel]

            return result

        # bookkeeping
        sum_preds = np.sum(self.model.predictions, axis=-1)
        square_norm_preds = np.linalg.norm(
            self.model.predictions, axis=-1, ord=2)**2

        # split data in batches
        split_indices = np.array_split(
            np.arange(self.data.shape[0]), n_batches)
        data_batches = np.array_split(self.data, n_batches, axis=0)
        if verbose:
            print("Each batch contains approx. " +
                  str(data_batches[0].shape[0]) + " voxels.")

        # perform grid fit
        grid_search_rbs = Parallel(self.n_jobs, verbose=verbose)(
            delayed(rsq_betas_for_batch)(
                data=data,
                vox_num=vox_num,
                predictions=self.model.predictions,
                n_timepoints=self.n_timepoints,
                data_var=self.data_var,
                sum_preds=sum_preds,
                square_norm_preds=square_norm_preds)
            for data, vox_num in zip(data_batches, split_indices))

        grid_search_rbs = np.concatenate(grid_search_rbs, axis=0)

        max_rsqs = grid_search_rbs[:, 0].astype('int')
        self.gridsearch_r2 = grid_search_rbs[:, 1]
        self.best_fitting_baseline = grid_search_rbs[:, 2]
        self.best_fitting_beta = grid_search_rbs[:, 3]

        # output
        self.gridsearch_params = np.array([
            self.model.vert_centres_flat[max_rsqs],
            self.model.sigmas_flat[max_rsqs],
            self.best_fitting_beta,
            self.best_fitting_baseline,
            self.gridsearch_r2
        ]).T
        
        # Put the vertex centres into a dictionary.
        self.vertex_centres=self.gridsearch_params[:,0].astype(int)
        self.vertex_centres_dict = [{'vert':k} for k in self.vertex_centres]
        
    def quick_grid_fit(self,sigma_grid):
        
        
        """quick_grid_fit
        Performs fast estimation of vertex centres and sizes using a simple dot product of zscored data.
        Does not complete the regression equation (estimating beta and baseline).
        Parameters
        ----------
        sigma_grid : 1D ndarray
            to be filled in by user
        Returns
        -------
        quick_gridsearch_params: An array containing the gridsearch parameters.
        quick_vertex_centres: An array containing the vertex centres.
        quick_vertex_centres_dict: A dictionary containing the vertex centres.
        """
        
        # Let the model create the timecourses
        self.model.create_grid_predictions(sigma_grid)
        
        self.model.predictions = self.model.predictions.astype('float32')        
        
        # Z-score everything so we can use dot product.
        zdat,zpreds=zscore(self.data.T),zscore(self.model.predictions.T)
        
        # Get all the dot products via np.tensordot.
        fits=np.tensordot(zdat,zpreds,axes=([0],[0]))
        
        # Get the maximum R2 and it's index. 
        max_rsqs,idxs = (np.amax(fits, 1)/zdat.shape[0])**2, np.argmax(fits, axis=1)
        
        self.idxs=idxs
        
        # Output centres, sizes, R2. 
        self.quick_gridsearch_params = np.array([
            self.model.vert_centres_flat[idxs].astype(int),
            self.model.sigmas_flat[idxs],
            max_rsqs]).T
        
        
        # We don't want to submit the vertex_centres for the iterative fitting - these are an additional argument.
        # Save them as .int as bundling them into an array with floats will change their type.
        self.quick_vertex_centres=self.quick_gridsearch_params[:,0].astype(int)
        
        # Bundle this into a dictionary so that we can use this as one of the **args in the iterative fitter
        self.quick_vertex_centres_dict = [{'vert':k} for k in self.quick_vertex_centres]
    
    def get_quick_grid_preds(self,dset='train'):
        
        """get_quick_grid_preds
        Returns the best fitting grid predictions from the quick_grid_fit method.
        Parameters
        ----------
        dset : Which dataset to return for (train or test).
        Returns
        -------
        train_predictions.
        OR
        test_predictions.
        """
        
        # Get the predictions of the best grid fits.
        # All we have to do is index the predictions via the index of the best-fitting prediction for each vertex.
        predictions=self.model.predictions[self.idxs,:]
        
        # Assign to object.
        if dset=='train':
            self.train_predictions=predictions
        elif dset=='test':
            self.test_predictions=predictions
    
    def quick_xval(self,test_data,test_stimulus):
        """quick_xval
        Takes the fitted parameters and tests their performance on the out of sample data.
        Parameters
        ----------
        Test data: Data to test predictions on.
        Test stimulus: CFstimulus class associated with test data.
        Returns
        -------
        CV_R2 - the out of sample performance.
        """
        
        fit_stimulus = deepcopy(self.model.stimulus) # Copy the test stimulus.
        self.test_data=test_data # Assign test data
        
        if test_stimulus is not None:    
            # Make the same grid predictions for the test data - therefore assign the new stimulus to the model class.
            self.model.stimulus = test_stimulus
        
        # Now we can generate the same test predictions with the test design matrix.
        self.model.create_grid_predictions(self.model.sigmas,'cart')
        
        # For each vertex, we then take the combination of parameters that provided the best fit to the training data.
        self.get_quick_grid_preds('test')
        
        # We can now put the fit stimulus back. 
        self.model.stimulus = fit_stimulus
        
        # Zscore the data and the preds
        zdat,zpred=zscore(self.test_data,axis=1),zscore(self.test_predictions,axis=1)

        def squaresign(vec):
            """squaresign
                Raises something to a power in a sign-sensive way.
                Useful for if dot products happen to be negative.
            """
            vec2 = (vec**2)*np.sign(vec)
            return vec2
        
        # Get the crossval R2. Here we use np.einsum to calculate the correlations across each row of the test data and the test predictions
        self.xval_R2=squaresign(np.einsum('ij,ij->i',zpred,zdat)/self.test_data.shape[-1])
