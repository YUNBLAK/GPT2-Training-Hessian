import numpy as np
import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from contextlib import nullcontext
import json
from scipy.linalg import block_diag
import sys
import os

# Add parent directory to path to import JS
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import JS


class Hessian(object): # AKA. MNISTHessian Parent Class
    def __init__(self, model = None,  m = 100, sigma = 1e-5**0.5, ckpt_iteration= 0, train_data = [], block_size = None, batch_size = None, num_v = 10, ctx =nullcontext(), use_minibatch = True, gradient_accumulation_steps = 1, device = 'cuda',  sample_layer = None, ddp = False, comment = None):
        self.model = model
        self.m = m # number of lanzcos basis
        self.sigma = sigma # the standard deviation of gaussian r.v.
        self.ckpt_iteration = ckpt_iteration
        self.train_data = train_data
        self.block_size = block_size
        self.batch_size = batch_size
        self.ctx = ctx
        self.use_minibatch = use_minibatch
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self.sample_layer = sample_layer
        self.ddp = ddp
        self.num_v = num_v
        self.num_bins = 1000


 
        total_elements = len(self.train_data)
        self.num_batches = total_elements // (self.batch_size * self.block_size)
        
        print('total batch', self.num_batches)

        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #n_params = sum(p.numel() for p in self.parameters())
        print('total params', self.total_params)
   

        self.comment = comment + '_minibatch_'+str(self.use_minibatch) +'_bs_'+str(self.batch_size*(self.gradient_accumulation_steps))+ '_m_'+str(self.m)  + '_v_' +str(self.num_v) + '_ckpt_'+str(self.ckpt_iteration)

        self.file_dir = 'files/'+str(self.comment)+'/'

        os.makedirs(self.file_dir, exist_ok= True)


    def get_spectrum(self, layer_by_layer = False):
        if layer_by_layer: 
            self.get_spectrum_layer_by_layer()
        else: 
            self.get_spectrum_full()


    def get_spectrum_layer_by_layer(self):
        weights_dic, values_dic, zhz_sum_dic, hz_sum_dic = {}, {}, {}, {}
        ctd_measure_list = []
        for name, param in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if param.requires_grad:

                zeros = np.zeros((self.num_v, self.m))
                weights_dic[name] = [row.tolist() for row in zeros]
                values_dic[name] =  [row.tolist() for row in zeros]

    
        t_s = time.time()
        for k in range(self.num_v): 
            print('current k' , k)

            'wiki version'
            T_dic, zhz_dic, w_prime_dic = self.tridiagonalize_by_lanzcos_layer_by_layer(k)[0], self.tridiagonalize_by_lanzcos_layer_by_layer(k)[1], self.tridiagonalize_by_lanzcos_layer_by_layer(k)[2] #returns a dic: {'name': T}

            # tridiagonalize with mutliple directions

            # for every direction we take the hadamard product z*Hz
            # and then we average over the directions
            for name, zhz_vec in zhz_dic.items():
                if name not in zhz_sum_dic:
                    zhz_sum_dic[name] = torch.zeros_like(zhz_vec).cpu()
                if name not in hz_sum_dic:
                    hz_sum_dic[name] = torch.zeros_like(w_prime_dic[name]).cpu()
                zhz_sum_dic[name] += zhz_vec.cpu() / self.num_v # sum and average over all directions
                hz_sum_dic[name] += w_prime_dic[name].cpu() / self.num_v # sum and average over all directions
            
            for name, T in T_dic.items():
                eigenvalues, U  = np.linalg.eigh(T)
                values_dic[name][k] = eigenvalues.tolist() #array to list
                weights_dic[name][k] = (U[0]**2).tolist()

                # calculate CTD measure for this layer
                numerator = torch.sum(zhz_sum_dic[name] ** 2)
                denominator = torch.sum(hz_sum_dic[name] ** 2)
                ctd_measure = 1 - (numerator/denominator).item()
                ctd_measure_list.append(ctd_measure)
                print(name, " CTD Measure: ", ctd_measure)
                print("numerator: ", numerator, "denominator: ", denominator)
            print("average CTD Measure: ", np.mean(ctd_measure_list))

            'we also save the inter-medium results'
            self.save_curve(total_time= time.time() - t_s, weights_layer = weights_dic, values_layer = values_dic)

        total_time = time.time() - t_s

        self.save_curve(total_time= total_time, weights_layer = weights_dic, values_layer = values_dic) # save curve for JS plot
        self._compute_and_save_block_stats(values_dic) # Save block statistics for add. info
        self._save_layer_eigenvalues(values_dic) # Save eigenvalues for JS plot
        self.visualize_hessian_matrix() # Plot block-diagonal Hessian heatmap
        JS.compute_js_blockwise(self.file_dir + 'layer_eigenvalues.json', out_png=self.file_dir + 'js_distance_blockwise_hessian.png') # Compute JS distance between layers


    def _save_layer_eigenvalues(self, values_layer_dict, fname="layer_eigenvalues.json"):
        out = {}
        for name, vals in values_layer_dict.items():
            flat = np.asarray(vals, dtype=np.float64).reshape(-1).tolist()
            out[name] = flat

        os.makedirs(self.file_dir, exist_ok=True)
        out_path = os.path.join(self.file_dir, fname)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    def get_spectrum_full(self):
        weights = np.zeros((self.num_v, self.m))
        values = np.zeros((self.num_v, self.m))
        time_initial = time.time()

        for k in range(self.num_v): 
            'wiki version'
            T = self.tridiagonalize_by_lanzcos(k)
            eigenvalues, U  = np.linalg.eigh(T)
            values[k,:] = eigenvalues
            weights[k,:] = U[0]**2
   

            self.save_curve(total_time = time.time() -time_initial, weights_full =  {'weights': weights}, values_full = {'values': values}, grid = [], curve = [])
            
        total_time = time.time() -time_initial
        grid, curve = self.interpolate(weights, values)

        self.save_curve(total_time = total_time, weights_full =  {'weights': weights}, values_full = {'values': values}, grid = grid, curve = curve)
        self._save_global_stats_from_values(values, fname="global_hessian_stats.json")

    def save_curve(self,total_time = None, weights_layer = None, values_layer = None, weights_full = None, values_full = None, grid = [], curve = []):
        if total_time != None:         
            file_name = self.file_dir + 'time.txt'
            with open(file_name, "w") as file:
                file.write(str(total_time) + "\n")

        if weights_layer != None:
            weights_layer = {key: weights_layer[key] for key in weights_layer} # convert the values to list
            file_name = self.file_dir + 'weights_layer.json'
            with open(file_name, 'w') as json_file:
                json.dump(weights_layer, json_file)
        

        if values_layer != None:
            values_layer = {key: values_layer[key] for key in values_layer} # convert the values to list
            file_name = self.file_dir + 'values_layer.json'
            with open(file_name, 'w') as json_file:
                json.dump(values_layer, json_file)

        if weights_full != None:
            weights_full = {key: weights_full[key].tolist() for key in weights_full} # convert the values to list
            file_name = self.file_dir + 'weights_full.json'
            with open(file_name, 'w') as json_file:
                json.dump(weights_full, json_file)
        

        if values_full != None:
            values_full = {key: values_full[key].tolist() for key in values_full} # convert the values to list
            file_name = self.file_dir + 'values_full.json'
            with open(file_name, 'w') as json_file:
                json.dump(values_full, json_file)


        if len(grid) != 0: 
            file_name = self.file_dir+ 'grid.txt'
            with open(file_name, "w") as file:
                for item in grid:
                    file.write(str(item) + "\n")

        if len(curve) != 0:
            file_name =  self.file_dir + 'curve.txt'
            with open(file_name, "w") as file:
                for item in curve:
                    file.write(str(item) + "\n")
        

    def load_curve(self, layer_by_layer = False):
        if layer_by_layer: 
            self.load_curve_layer_by_layer()
        else: 
            self.load_curve_full()

      
    def load_curve_layer_by_layer(self):
        'load weights and values:'
        file_name = self.file_dir + 'weights_layer.json'
        with open(file_name, 'r') as json_file:
            weights_dic = json.load(json_file)
        weights_dic = {key: np.array(value) for key, value in weights_dic.items()}


        file_name = self.file_dir + 'values_layer.json'
        with open(file_name, 'r') as json_file:
            values_dic = json.load(json_file)
        values_dic = {key: np.array(value) for key, value in values_dic.items()}

        for name in weights_dic.keys():
            weights = weights_dic[name]
            values = values_dic[name]
            grid, curve = self.interpolate(weights, values)

            # Normalize by 10th largest eigenvalue
            all_eigenvalues = values.flatten()
            sorted_eigenvalues = np.sort(all_eigenvalues)[::-1]  # Sort descending
            if len(sorted_eigenvalues) >= 10:
                lambda_10 = sorted_eigenvalues[9]  # 10th largest (0-indexed: 9)
            else:
                # If we have fewer than 10 eigenvalues, use the largest
                lambda_10 = sorted_eigenvalues[0] if len(sorted_eigenvalues) > 0 else 1.0
                print(f'[WARN] Layer {name} has only {len(sorted_eigenvalues)} eigenvalues, using largest for normalization')
            
            # Normalize grid by 10th largest eigenvalue
            grid_normalized = np.array(grid) / lambda_10
            
            print(f'Layer {name}: 10th largest eigenvalue = {lambda_10:.6e}')
            print('curve',curve)
            'plot'
            plt.figure()
            plt.plot(grid_normalized, curve, label = 'approximated curve', alpha = 0.5)
            plt.xlabel('Eigenvalues normalized')
            plt.ylabel('Frequency')
            plt.ylim([1e-10,1e2])
            plt.legend()
            plt.title(f'model at iteration {self.ckpt_iteration} (normalized by 10th largest eigenvalue)')
            plt.savefig(self.file_dir+'spectrum_'+name+'.png')
            plt.close()

            'log plot'
            plt.figure()
            plt.semilogy(grid_normalized, curve, label = 'approximated curve', alpha = 0.5)
            plt.xlabel('Eigenvalues normalized')
            plt.ylabel('Frequency (log)')
            plt.ylim([1e-10,1e2])
            # Extend x-axis to show negative values if they exist
            x_min = min(np.min(grid_normalized), -0.2)
            x_max = max(np.max(grid_normalized), 1.2)
            plt.xlim([x_min, x_max])
            plt.legend()
            plt.title(f'model at iteration {self.ckpt_iteration} (normalized by 10th largest eigenvalue)')
            plt.savefig(self.file_dir+'/spectrum_log_'+name+'.png')
            plt.close()


    def load_curve_full(self):
        'load curve'
        grid = []
        file_name = self.file_dir + 'grid.txt'
        with open(file_name, "r") as file:
            for line in file:
                grid.append(float(line.strip()))  # Use strip() to remove 

        file_name =  self.file_dir + 'curve.txt'
        curve = []
        with open(file_name, "r") as file:
            for line in file:
                curve.append(float(line.strip()))  # Use strip() to remove 

        # Load eigenvalues to compute 10th largest for normalization
        lambda_10 = 1.0  # Default if we can't load values
        try:
            file_name = self.file_dir + 'values_full.json'
            with open(file_name, 'r') as json_file:
                values_full = json.load(json_file)
                if 'values' in values_full:
                    values = np.array(values_full['values'])
                    all_eigenvalues = values.flatten()
                    sorted_eigenvalues = np.sort(all_eigenvalues)[::-1]  # Sort descending
                    if len(sorted_eigenvalues) >= 10:
                        lambda_10 = sorted_eigenvalues[9]  # 10th largest
                    elif len(sorted_eigenvalues) > 0:
                        lambda_10 = sorted_eigenvalues[0]
                        print(f'[WARN] Full Hessian has only {len(sorted_eigenvalues)} eigenvalues, using largest for normalization')
                    print(f'Full Hessian: 10th largest eigenvalue = {lambda_10:.6e}')
        except (FileNotFoundError, KeyError) as e:
            print(f'[WARN] Could not load values_full.json for normalization: {e}')
        
        # Normalize grid by 10th largest eigenvalue
        grid_normalized = np.array(grid) / lambda_10

        'plot'
        plt.figure()
        plt.plot(grid_normalized, curve, label = 'approximated curve', alpha = 0.5)
        plt.xlabel('Eigenvalues normalized')
        plt.ylabel('Frequency')
        plt.ylim([1e-10,1e2])
        # plt.xlim([-5, 5])
        plt.legend()
        plt.title(f'model at interation {self.ckpt_iteration} (normalized by 10th largest eigenvalue)')
        plt.savefig(self.file_dir+'/spectrum_full_hessian.png')
        plt.close()

        'log plot'
        plt.figure()
        plt.semilogy(grid_normalized, curve, label = 'approximated curve', alpha = 0.5)
        plt.xlabel('Eigenvalues normalized')
        plt.ylabel('Frequency (log)')
        plt.ylim([1e-10,1e2])
        #plt.xlim([3, 5])
        plt.legend()
        plt.title(f'model at interation {self.ckpt_iteration} (normalized by 10th largest eigenvalue)')
        plt.savefig(self.file_dir+'/spectrum_log_full_hessian.png')
        plt.close()


    def tridiagonalize_by_lanzcos_layer_by_layer(self, k):
        v_dic = {} # value: list
        alpha_dic = {} # value: scaler
        w_dic = {} # value: #parameters*1 tensor
        beta_dic = {} # value: scaler
        T_dic = {} # value: m*m tensor 
        zhz_dic = {} # value: # for every layer z*Hz, store this vector
        'initialize'
        for name, params in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if params.requires_grad:
                v = torch.randn_like(params, dtype = torch.float32) 
                v /= torch.norm(v)
                v_dic[name] = [v.cpu()]
                T_dic[name] = np.zeros((self.m, self.m), dtype= np.float64)


        w_prime_dic = self.hessian_vector_product_with_dic_input(v_dic, k,0)

        'orthogonalize wprime'
        for name in T_dic.keys():
            alpha_dic[name] = torch.sum(w_prime_dic[name] * v_dic[name][-1])  
            w_dic[name] = w_prime_dic[name] - alpha_dic[name] * v_dic[name][-1]
            T_dic[name][0, 0] = alpha_dic[name] 

        'iteration'
        print('runing lanczos')
        for j in range(1, self.m):
            for name in T_dic.keys(): 
                beta = torch.norm(w_dic[name])
                beta_dic[name] = beta
                if beta >1e-8:
                    v_dic[name].append( w_dic[name] / beta )
                else:
                    #print('The value of beta is 0')
                    v_dic[name].append( w_dic[name] / 1e-8 )
                    #raise ZeroDivisionError('The value of beta is 0')
                if len(v_dic[name]) > 2:
                    del v_dic[name][0]  # keep this list short to save memory

            t_hessian = time.time()
  
            w_prime_dic = self.hessian_vector_product_with_dic_input(v_dic, k,j)
            print('t for hessian', time.time() - t_hessian)

            'orthogonalize wprime'
            for name in T_dic.keys():
                alpha_dic[name] = torch.sum(w_prime_dic[name] * v_dic[name][-1])  
                w_dic[name] = w_prime_dic[name] - alpha_dic[name] * v_dic[name][-1] - beta_dic[name] * v_dic[name][-2]
                T_dic[name][j, j] = alpha_dic[name] 
                T_dic[name][j-1, j ] = beta_dic[name] 
                T_dic[name][j , j-1] = beta_dic[name]

        'store z*Hz for every layer'
        for name in T_dic.keys():
            zhz_dic[name] = v_dic[name][-1] * w_prime_dic[name]

        return  T_dic, zhz_dic, w_prime_dic


    def tridiagonalize_by_lanzcos(self, k):
        'set up'
        v_list = []
        T = np.zeros((self.m, self.m), dtype= np.float64)

        'initialization'
        # Use float32 for MPS compatibility, running locally
        v = torch.randn(self.total_params, dtype = torch.float32) 
        v /= torch.norm(v)
        v_list.append(v.cpu())


        w_prime = self.hessian_vector_product_with_tensor_input(v_list[-1], k,0)
        'orthogonalize wprime'
        alpha = torch.sum(w_prime * v_list[-1])
        w = w_prime - alpha * v_list[-1]
        T[0, 0] = alpha

        'iteration'
        #t_s = time.time()
        print('runing lanczos')
        for j in range(1, self.m):
            beta = torch.norm(w)
            if beta >1e-8:
                v_list.append(w / beta)

            else:
                v_list.append(w / 1e-8)

                # print(f' since beta = {beta}, generate v that orthogonal to all previous v')
                # # Generate a random vector orthogonal to previous ones
                # v = torch.randn(self.total_params) *(1/self.total_params)**0.5
                # for i in range(j):
                #     vi = v_list[i]
                #     v -= torch.sum(vi * v) * vi
                # v /= torch.norm(v)
                if len(v_list) > 2:
                    del v_list[0]  # keep this list short to save memory


            w_prime = self.hessian_vector_product_with_tensor_input(v_list[-1], k,j)
            alpha = torch.sum(w_prime* v_list[-1])
            w = w_prime - alpha * v_list[-1] - beta * v_list[-2]
            T[j, j] = alpha
            T[j-1, j ] = beta
            T[j , j-1] = beta
         
        return  T


    def interpolate(self,weights, values):
        left_boundary = np.mean(np.min(values, axis = 1))-1
        right_boundary= np.mean(np.max(values, axis = 1)) +1
        n_grid = 50000
        grid = np.linspace(left_boundary, right_boundary, n_grid).tolist()
        density_all = np.zeros((self.num_v, n_grid))

        for k  in range(self.num_v):
            for idx, t  in enumerate(grid):
                values_each_v_t = self.gaussian_density(t, values[k,:])
                density_each_v_t = np.sum(values_each_v_t * weights[k,:])
                density_all[k,idx] = density_each_v_t

        density_avg = np.nanmean(density_all, axis = 0)
        norm_fact = np.sum(density_avg)*(grid[1]- grid[0])
        density_avg /= norm_fact

        return grid, density_avg
 
    def _save_global_stats_from_values(self, values, fname="global_hessian_stats.json"):
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        h_max = float(np.max(flat))
        h_min = float(np.min(flat))
        h_var = float(np.var(flat))
        cond = float(h_max / h_min) if h_min > 0 else float("inf")
        out = {"h_max": h_max, "h_min": h_min, "h_variance": h_var, "condition_number": cond}
        os.makedirs(self.file_dir, exist_ok=True)
        with open(os.path.join(self.file_dir, fname), "w") as f:
            json.dump(out, f, indent=2)


    def _compute_and_save_block_stats(self, values_layer_dict):

        block_stats = {}
        for name, vals in values_layer_dict.items():
            flat = np.asarray(vals).astype(np.float64).reshape(-1)
            h_max = float(np.max(flat))
            h_min = float(np.min(flat))
            h_var = float(np.var(flat))
            if h_min > 0:
                cond = float(h_max / h_min)
            else:
                cond = float("inf")

            block_stats[name] = {
                "h_max": h_max,
                "h_min": h_min,
                "h_variance": h_var,
                "condition_number": cond,
            }
        print("[Block-wise Hessian stats]")
        for name, st in block_stats.items():
            print(f" - {name}: h_max={st['h_max']:.4e}, h_min={st['h_min']:.4e}, "
                  f"h_var={st['h_variance']:.4e}, cond={st['condition_number']:.4e}")

        os.makedirs(self.file_dir, exist_ok=True)
        out_path = os.path.join(self.file_dir, "block_hessian_stats.json")
        with open(out_path, "w") as f:
            json.dump(block_stats, f, indent=2)

    def hessian_vector_product_with_dic_input(self, d_dic, v_step, l_step):
        'comput hessian_vector product, takes a dictionary as input, the values of dic is a list of historical lanscoz directions: d_dic = {name, [history v..]}'
        self.model.eval()
        self.model.zero_grad(set_to_none = True)

        'initialize'
        hd_dic = {}
        for name, param in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if param.requires_grad:
                hd_dic[name]  = torch.zeros_like(param.data).cpu()


        t_hd = time.time()
        for batch_idx in range(self.num_batches):

            
            X, Y = self.get_batch(batch_idx)
            with self.ctx:
                _, loss = self.model(X, Y)

            loss.backward(create_graph= True)
            g_dic = {}
            for name, param in self.model.named_parameters():
                if name not in self.sample_layer:
                    continue
                if param.requires_grad:
                    # Use float32 for MPS compatibility, running locally
                    g_dic[name] = param.grad.float()

        
            self.model.zero_grad(set_to_none = True)
            for name, param in self.model.named_parameters():
                if name not in self.sample_layer:
                    continue
                if param.requires_grad:
                    l = torch.sum(g_dic[name].to(self.device) * d_dic[name][-1].to(self.device))
                    l.backward(retain_graph = True)

                    # Use float32 for MPS compatibility, running locally
                    hd = param.grad.float().data.clone()
                    hd_dic[name]  += hd.cpu() 
                    self.model.zero_grad(set_to_none = True)

            if batch_idx % 10 == 1 or batch_idx == self.gradient_accumulation_steps-1:
                print(f'layer hessian: load_iter ={self.ckpt_iteration}, current random direction = {v_step} / {self.num_v}, lanczos step = {l_step} / {self.m}, Hd current batch = {batch_idx} / {self.num_batches}, time = {time.time() -t_hd}')
                t_hd = time.time()

            if self.use_minibatch == True and batch_idx == self.gradient_accumulation_steps-1:
                break
        return hd_dic

    def hessian_vector_product_with_tensor_input(self, d_tensor, v_step, l_step):
        """ Comput hessian_vector product, takes a flattened tensors as input (with shape (total parameters, ) ) """
        d_tensor = d_tensor.to(self.device)
        self.model.eval()
        self.model.zero_grad(set_to_none = True)
        total_hd_tensor = 0

        t_hd = time.time()
        for batch_idx in range(self.num_batches):
            X, Y = self.get_batch(batch_idx)
            with self.ctx:
                _, loss = self.model(X, Y)

            loss.backward(create_graph= True)
            g_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    g_list.append(torch.flatten(param.grad.float()))

            g_tensor = torch.cat(g_list, dim = 0)
            
            self.model.zero_grad(set_to_none = True)
            g_tensor = g_tensor.to(self.device)
            l = torch.sum(g_tensor*d_tensor)
            l.backward(retain_graph = True)

            hd_list = []
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    hd_list.append(torch.flatten(param.grad.float().data.clone()))

            hd_tensor = torch.cat(hd_list, dim = 0)
            self.model.zero_grad(set_to_none = True)
            hd_tensor = hd_tensor.cpu()
            total_hd_tensor += hd_tensor

            if batch_idx % 10 == 1 or batch_idx == self.gradient_accumulation_steps-1:
                print(f'full hessian: load_iter ={self.ckpt_iteration} current random direction = {v_step} / {self.num_v}, lanczos step = {l_step} / {self.m}, Hd current batch = {batch_idx} / {self.num_batches}, time = {time.time() -t_hd}')
                t_hd = time.time()

            if self.use_minibatch == True and batch_idx == self.gradient_accumulation_steps-1:
                break
        return total_hd_tensor

    def get_batch(self, batch_idx):
        start_idx = batch_idx * self.batch_size * self.block_size
        end_idx = (batch_idx + 1) * self.batch_size * self.block_size
        X = torch.from_numpy((self.train_data[start_idx:end_idx]).astype(np.int64)).reshape(self.batch_size, self.block_size)
        Y = torch.from_numpy((self.train_data[start_idx+1:end_idx+1]).astype(np.int64)).reshape(self.batch_size, self.block_size)
        # pin_memory only works with CUDA, skip for MPS/CPU
        if self.device.startswith('cuda'):
            X, Y = X.pin_memory().to(self.device, non_blocking=True), Y.pin_memory().to(self.device, non_blocking=True)
        else:
            X, Y = X.to(self.device), Y.to(self.device)
        return X, Y


    def get_true_curve(self, grid, eigenvalues):
        curve = []
        for t in grid:
            density = self.gaussian_density(t, eigenvalues)
            value = np.mean(density)
            curve.append(value)
        return curve
        

    def gaussian_density(self, t, values):
        coeff = 1.0 / np.sqrt(2 * math.pi * self.sigma**2)
        val = -(values - t) ** 2
        val = val / (2.0 * self.sigma**2)
        val = np.exp(val)
        density = coeff * val
        return density

    def visualize_hessian_matrix(self):
        """ Visualize the Hessian matrix as a heatmap using Lanczos approximation """
        
        # Run one Lanczos iteration to get tridiagonal matrices and vectors
        k = 0  # Fix direction to 1st
        T_dic, V_dic, ctd_heatmap_iter_over_layers = self.tridiagonalize_by_lanzcos_layer_by_layer_with_vectors(k)[0], self.tridiagonalize_by_lanzcos_layer_by_layer_with_vectors(k)[1], self.tridiagonalize_by_lanzcos_layer_by_layer_with_vectors(k)[2]

        # Reconstruct Hessian approximations for each layer
        H_approx_dic = {}
        param_sizes = {}
        
        for name in T_dic.keys():
            # per-layer H reconstruction apprixmation: H ≈ V T V^T
            T = T_dic[name]
            V_list = V_dic[name] # lanzcos vectors basis
            
            if len(V_list) > 0:
                # Get parameter size
                param = next(p for n, p in self.model.named_parameters() if n == name and p.requires_grad)
                param_size = param.numel()
                
                # Stack Lanczos vectors into V (param_size × m)
                V = torch.stack([v.flatten() for v in V_list], dim=1).cpu().numpy()
                
                H_approx = V @ T @ V.T
                
                H_approx_dic[name] = H_approx
                param_sizes[name] = param_size
                print(f"Layer {name}: Reconstructed {param_size}×{param_size} Hessian from {len(V_list)} Lanczos vectors")
        
        # Compute global scale from ALL layers for scale norm
        all_abs_values = []
        for H in H_approx_dic.values():
            all_abs_values.extend(np.abs(H).flatten())
        all_abs_values = np.array(all_abs_values)
        global_vmax = np.percentile(all_abs_values[all_abs_values > 0], 95)
        print(f"[INFO] Global visualization scale (95th percentile): {global_vmax:.6e}")

        print(f"[INFO] CTD Measure Heatmap: {ctd_heatmap_iter_over_layers.shape}")
        layer_names = list(T_dic.keys())
        plt.figure(figsize=(10, 6))
        plt.imshow(ctd_heatmap_iter_over_layers.cpu().numpy(), cmap='viridis', aspect='auto')
        plt.yticks(range(len(layer_names)), layer_names)
        plt.colorbar()
        plt.title(f'CTD Measure Heatmap (Lanczos, m={self.m})')
        plt.xlabel('Lanczos Step')
        plt.ylabel('Layer')
        plt.tight_layout()
        plt.savefig(self.file_dir + '/ctd_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create block-diagonal visualization
        self._plot_hessian_heatmap(H_approx_dic, param_sizes, global_vmax)
        
        # Also plot individual layer Hessians
        for name, H in H_approx_dic.items():
            self._plot_single_layer_hessian(name, H, global_vmax)

    def tridiagonalize_by_lanzcos_layer_by_layer_with_vectors(self, k):
        """
        Lanczos method that also returns the Lanczos vectors V for Hessian reconstruction.
        Returns: (T_dic, V_dic) where V_dic contains all Lanczos vectors (not just last 2).
        """
        v_dic = {}  # Store ALL vectors for reconstruction
        alpha_dic = {}
        w_dic = {}
        beta_dic = {}
        T_dic = {}
        
        ' initialize '
        for name, params in self.model.named_parameters():
            if name not in self.sample_layer:
                continue
            if params.requires_grad:
                v = torch.randn_like(params, dtype=torch.float32)
                v /= torch.norm(v)
                v_dic[name] = [v.to(self.device)]  # Store all vectors on device
                T_dic[name] = np.zeros((self.m, self.m), dtype=np.float64)

        num_layers = len(T_dic.keys())
        ctd_heatmap_iter_over_layers = torch.zeros((num_layers, self.m), dtype=torch.float32)

        w_prime_dic = self.hessian_vector_product_with_dic_input(v_dic, k, 0)
        
        # Move w_prime_dic to the correct device
        for name in w_prime_dic.keys():
            w_prime_dic[name] = w_prime_dic[name].to(self.device)

        ' orthogonalize wprime '
        for name in T_dic.keys():
            v_dic[name][-1] = v_dic[name][-1].to(self.device) # Ensure v_dic vectors are on the correct device
            alpha_dic[name] = torch.sum(w_prime_dic[name] * v_dic[name][-1])
            w_dic[name] = w_prime_dic[name] - alpha_dic[name] * v_dic[name][-1]
            T_dic[name][0, 0] = alpha_dic[name]

        ' iteration'
        print('Running Lanczos for Hessian visualization...')
        for j in range(1, self.m):
            # loops for each probe vector 
            for layer_idx, name in enumerate(T_dic.keys()):
                beta = torch.norm(w_dic[name])
                beta_dic[name] = beta
                if beta > 1e-8:
                    v_new = w_dic[name] / beta
                else:
                    v_new = w_dic[name] / 1e-8
                
                'calculate CTD measure for this layer'
                numerator_ctd = torch.sum((v_dic[name][-1] * w_prime_dic[name]).float() ** 2)
                denominator_ctd = torch.sum((w_prime_dic[name]).float() ** 2)
                ctd_heatmap_iter_over_layers[layer_idx, j] = (1 - (numerator_ctd / denominator_ctd)).item()

                # Store ALL vectors (for reconstruction)
                v_dic[name].append(v_new.to(self.device))

            t_hessian = time.time()
            w_prime_dic = self.hessian_vector_product_with_dic_input(v_dic, k, j)

            for name in w_prime_dic.keys():
                w_prime_dic[name] = w_prime_dic[name].to(self.device) # Move w_prime_dic to the correct device to match v_dic
            print(f'Lanczos step {j}/{self.m}, time: {time.time() - t_hessian:.2f}s')


            # Orthogonalize wprime
            for name in T_dic.keys():
                alpha_dic[name] = torch.sum(w_prime_dic[name] * v_dic[name][-1])
                w_dic[name] = w_prime_dic[name] - alpha_dic[name] * v_dic[name][-1] - beta_dic[name] * v_dic[name][-2]
                T_dic[name][j, j] = alpha_dic[name]
                T_dic[name][j - 1, j] = beta_dic[name]
                T_dic[name][j, j - 1] = beta_dic[name]

        return T_dic, v_dic, ctd_heatmap_iter_over_layers

    def _plot_hessian_heatmap(self, H_approx_dic, param_sizes, global_vmax):
        'Create a block-diagonal heatmap visualization of the Hessian'

        if not H_approx_dic:
            print("[WARN] No Hessian approximations to visualize")
            return
        max_display_size = 1e10  # Maximum size to display
        
        # Create block-diagonal structure
        total_size = sum(param_sizes.values())
        
        if total_size > max_display_size:
            print(f"[INFO] Total Hessian size ({total_size}) too large, creating downsampled visualization")
            H_blocks = [] # Downsample each block
            for name in sorted(H_approx_dic.keys()): 
                H = H_approx_dic[name]
                if H.shape[0] > max_display_size // len(H_approx_dic): # Downsample by taking every nth element
                    step = max(1, H.shape[0] // (max_display_size // len(H_approx_dic)))
                    H = H[::step, ::step]
                H_blocks.append(H)
            
            'Create partial block diagonal'
            if block_diag is not None and len(H_blocks) > 1:
                H_block_diag = block_diag(*H_blocks)
            else:
                H_block_diag = H_blocks[0] if len(H_blocks) == 1 else self._manual_block_diag(H_blocks)
        else:
            'Create full block diagonal'
            H_blocks = [H_approx_dic[name] for name in sorted(H_approx_dic.keys())]
            if block_diag is not None and len(H_blocks) > 1:
                H_block_diag = block_diag(*H_blocks)
            else:
                H_block_diag = self._manual_block_diag(H_blocks) # fallback to manual block diag
        
        # Use global scale
        vmax = global_vmax
        vmin = 0
        
        'Plot'
        plt.figure(figsize=(12, 10))
        plt.imshow(H_block_diag, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Hessian value')
        plt.title(f'Hessian Matrix Approximation (Lanczos, m={self.m})\nBlock-diagonal structure by layer')
        plt.xlabel('Parameter index')
        plt.ylabel('Parameter index')
        plt.tight_layout()
        plt.savefig(self.file_dir + '/hessian_matrix_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved block-diagonal Hessian heatmap to {self.file_dir}/hessian_matrix_heatmap.png")

    def _plot_single_layer_hessian(self, name, H, global_vmax):
        'Plot individual layer Hessian as heatmap'

        vmax = global_vmax
        vmin = 0
        
        'Plot'
        plt.figure(figsize=(10, 8))
        plt.imshow(H, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Hessian value')
        plt.title(f'Hessian Matrix: {name}\n(Lanczos approximation, m={self.m})')
        plt.xlabel('Parameter index')
        plt.ylabel('Parameter index')
        plt.tight_layout()
        safe_name = name.replace('.', '_').replace('/', '_')
        plt.savefig(self.file_dir + f'/hessian_matrix_{safe_name}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved layer Hessian heatmap: {safe_name}")

    def _manual_block_diag(self, blocks):
        'Manually create block diagonal matrix from list of blocks'

        total_size = sum(b.shape[0] for b in blocks)
        H_block_diag = np.zeros((total_size, total_size))
        start_idx = 0
        for H in blocks:
            end_idx = start_idx + H.shape[0]
            H_block_diag[start_idx:end_idx, start_idx:end_idx] = H
            start_idx = end_idx
        return H_block_diag