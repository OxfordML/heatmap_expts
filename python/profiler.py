import cProfile
import gen_synthetic
command = 'gen_synthetic.run_experiments()'
cProfile.run( command, filename="../gen_synthetic.profile") 