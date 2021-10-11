from pycondor import Job 
import os
base_out_dir = '/home/srashti.goyal/lensid_runs/trial'
job_name = 'o3_condor_trial'
executable = 'ml_predict_workflow.py'
arguments = None
submit_job = 1
exec_file_loc = '/home/srashti.goyal/.conda/envs/igwn-py37-hanabi/bin/'
accounting_tag = 'ligo.prod.o3.cbc.testgr.tiger'
error = os.path.abspath(base_out_dir + '/condor/error')
output = os.path.abspath(base_out_dir + '/condor/output')
log = os.path.abspath(base_out_dir + '/condor/log')
submit = os.path.abspath(base_out_dir + '/condor/submit')
    
job = Job(name=job_name,
          executable=executable,
          submit=submit, 
          error=error, 
          output=output,
          log=log,
          arguments=arguments,
          universe='vanilla',
          getenv=True,
          extra_lines=['accounting_group = ' + accounting_tag],
          request_memory='4GB')

if submit_job ==1:       
    '  submitting the job...'
    job.build_submit()
        
else:
    job.build()
    print('\n \n job saved at: %s'%(submit +'/' + job.name))