
(defun comment-or-uncomment-region-or-line ()
    "Comments or uncomments the region or the current line if
there's no active region."
    (interactive)
    (let (beg end)
        (if (region-active-p)
            (setq beg (region-beginning) end (region-end))
            (setq beg (line-beginning-position) end (line-end-position)))
        (comment-or-uncomment-region beg end)))

(global-set-key (kbd "M-#") 'comment-or-uncomment-region-or-line)



(defun name-ss-ext ()
  "Return the name of the file without the extension."
  (file-name-sans-extension (buffer-file-name))
)

(defun run-pdflatex-file-buffer ()
  "Executes 'pdflatex file-buffer' in inferior shell."
  (interactive)
  (save-buffer)
  (shell-command (concat "pdflatex -shell-escape \"" (name-ss-ext) ".tex\""))
  (setq buffer_tex (current-buffer))
  (pop-to-buffer "*Shell Command Output*")
  (goto-char (point-max)) 
  (recenter-top-bottom -2)
  (pop-to-buffer buffer_tex)
)

(defun run-latex-file-buffer ()
  "Executes 'latex file-buffer' in inferior shell."
  (interactive)
  (save-buffer)
  (shell-command (concat "latex \"" (name-ss-ext) "\".tex"))
  (setq buffer_tex (current-buffer))
  (pop-to-buffer "*Shell Command Output*")
  (goto-char (point-max)) 
  (recenter-top-bottom -2)
  (pop-to-buffer buffer_tex)
)


(defun run-dvipdf-file-buffer ()
  "Executes 'dvipdf file-buffer'."
  (interactive)
  ;; (shell-command (concat "dvipdf " (name-ss-ext)))
  (start-process-shell-command 
   "dvipdf" 
   "Messages"
   (concat "dvipdf " (name-ss-ext))
   )
)


(defun run-viewpdf-file-buffer ()
  "Executes 'viewpdf file-buffer'."
  (interactive)
  (start-process-shell-command 
   "pdfviewer" 
   "Messages_evince"
   ;; (buffer-name)
   (concat "evince " (name-ss-ext) ".pdf")
   )
  )


(defun run-bibtex-file-buffer ()
  "Executes 'bibtex file-buffer' in inferior shell."
  (interactive)
  (shell-command (concat "bibtex " (name-ss-ext)))
)



(global-set-key (kbd "C-c C-<f1>") 'run-viewpdf-file-buffer)
(global-set-key (kbd "C-c C-<f2>") 'run-pdflatex-file-buffer)
(global-set-key (kbd "C-c C-<f3>") 'run-latex-file-buffer)
(global-set-key (kbd "C-c C-<f4>") 'run-dvipdf-file-buffer)
(global-set-key (kbd "C-c C-<f5>") 'run-bibtex-file-buffer)


