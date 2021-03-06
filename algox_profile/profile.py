
import subprocess, tempfile, time

make_executable = "python3 make-exact.py"
executables = {
	"grf": "python3 solve-exact-grf.py",
	"libdlx": "python2 solve-exact-libdlx.py",
	"dlx-cpp": "dlx-cpp/build/dlx -pvs",
}

N, m, s = 100, 10, 1500
problem_file = tempfile.TemporaryFile("w+")
args = make_executable.split() + [str(arg) for arg in (N, m, s)]
subprocess.run(args, stdout = problem_file)
print("Running...")
for proj_name, exe in executables.items():
	problem_file.seek(0)
	args = exe.split()
	t0 = time.time()
#	subprocess.run(args, stdin = problem_file, stdout = subprocess.PIPE)
	subprocess.run(args, stdin = problem_file)
	dt = time.time() - t0
	print(proj_name, dt)

