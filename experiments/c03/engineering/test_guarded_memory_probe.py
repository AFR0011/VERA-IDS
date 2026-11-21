"""Supervisor checks: no benchmark observations or model fitting."""
import subprocess
import sys
import unittest
from unittest.mock import patch, Mock

import psutil
import guarded_memory_probe as guard


class SupervisorTests(unittest.TestCase):
    def test_tree_memory_includes_descendants(self):
        root, child = Mock(), Mock()
        root.children.return_value = [child]
        root.memory_info.return_value.rss = 10
        child.memory_info.return_value.rss = 100
        with patch.object(guard.psutil, 'Process', return_value=root):
            self.assertEqual(guard.tree_rss(123), 110)

    def test_monitor_error_is_not_zero_memory(self):
        with patch.object(guard.psutil, 'Process', side_effect=psutil.AccessDenied(123)):
            with self.assertRaises(psutil.AccessDenied):
                guard.tree_rss(123)

    def test_low_memory_never_launches_worker(self):
        with patch.object(guard.original, 'precheck_environment'), \
             patch.object(guard.original, 'os_memory', return_value={'physical_available':guard.GIB}), \
             patch.object(guard.subprocess, 'Popen') as launch:
            with self.assertRaises(RuntimeError):
                guard.supervise()
            launch.assert_not_called()

    def test_actual_python_tree_measurement_and_cleanup(self):
        child = subprocess.Popen([sys.executable, '-B', '-u', '-c',
            'import os,time; data=bytearray(32*1024*1024); print(os.getpid(),flush=True); time.sleep(30)'],
            stdout=subprocess.PIPE, text=True)
        try:
            worker_pid = int(child.stdout.readline())
            self.assertGreaterEqual(guard.tree_rss(child.pid), 32*1024*1024)
            guard.stop_tree(child.pid)
            child.wait(timeout=5)
            self.assertFalse(psutil.pid_exists(worker_pid))
        finally:
            if child.poll() is None:
                guard.stop_tree(child.pid)
                child.wait(timeout=5)
            child.stdout.close()

    def test_exceeded_memory_limits_stop_tree(self):
        for free, rss in [(guard.GIB, 1), (4*guard.GIB, 4*guard.GIB)]:
            with self.subTest(free=free, rss=rss):
                process = Mock(pid=123, returncode=-9)
                process.poll.return_value = None
                with patch.object(guard.original, 'precheck_environment'), \
                     patch.object(guard.original, 'os_memory', side_effect=[
                         {'physical_available':4*guard.GIB}, {'physical_available':free}]), \
                     patch.object(guard.subprocess, 'Popen', return_value=process), \
                     patch.object(guard, 'tree_rss', return_value=rss), \
                     patch.object(guard, 'stop_tree') as stop:
                    self.assertEqual(guard.supervise(), 3)
                    stop.assert_called_once_with(123)
                    process.wait.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
