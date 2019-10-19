
(set-language-environment "UTF-8")

(setq custom-file "~/.emacs.d/init-custom.el")
(load "~/.emacs.d/init-custom.el")
(load "~/.emacs.d/user_functions.el")

;; To use cask (https://github.com/cask/cask)
(require 'cask "~/.cask/cask.el")
(cask-initialize)


(column-number-mode t)
(display-time-mode t)

(load-library "iso-transl")

;; Spell check
(setq-default ispell-program-name "aspell")
(setq ispell-dictionary '"english")

(add-hook 'text-mode-hook 'flyspell-mode)
(add-hook 'prog-mode-hook 'flyspell-prog-mode)


;; Define some key bindings:
(global-set-key (kbd "C-c r") 'replace-string)

(global-set-key [(meta f3)] (make-hippie-expand-function
                             '(try-expand-dabbrev-visible
                               try-expand-dabbrev
                               try-expand-dabbrev-all-buffers) t))


;; link between emacs kill and copy in other programs
(setq x-select-enable-clipboard t)

;; this line causes cut/paste errors under windows (emacs 24)
(if (eq system-type 'gnu-linux) (setq interprogram-paste-function 'x-cut-buffer-or-selection-value) )
;; this alternative causes errors under linux (emacs 24)
(if (eq system-type 'windows-nt) (setq interprogram-paste-function 'x-selection-value) )


;; Modes for some type of files:
(setq auto-mode-alist (cons '("\\.txt$" . rst-mode) auto-mode-alist))

(load "~/.emacs.d/digiflow-mode.el")
(setq auto-mode-alist (cons '("\\.dfc$" . digiflow-mode) auto-mode-alist))

;; For Cython:
(load "~/.emacs.d/cython-mode.el")
(setq auto-mode-alist (cons '("\\.pyx$" . cython-mode) auto-mode-alist))
(setq auto-mode-alist (cons '("\\.pxd$" . cython-mode) auto-mode-alist))


;; Display settings:
(setq frame-background-mode 'dark)

(global-set-key (kbd "C-c w")   'whitespace-mode)
(setq whitespace-style
  '(face lines-tail
    tabs
    spaces
    ;; space-mark
    tab-mark)
)
(require 'whitespace)
(global-whitespace-mode 1)

(if (window-system) (set-frame-size (selected-frame) 166 40))

(add-to-list 'load-path "~/.emacs.d/color-theme-6.6.0")
(require 'color-theme)
(eval-after-load "color-theme"
  '(progn
     (color-theme-initialize)
     (color-theme-hober)))


;; Better buffer name style for files with the same names
(require 'uniquify)
(setq uniquify-buffer-name-style 'reverse)


;; For Latex:
(eval-after-load "tex"
  '(add-to-list 'TeX-command-list
    '("Pdflatex" "pdflatex %s" TeX-run-command t t :help "Run pdflatex") t))

(setq TeX-auto-save t)
(setq TeX-parse-self t)
(setq-default TeX-master nil)

(add-hook 'LaTeX-mode-hook 'visual-line-mode)
(add-hook 'LaTeX-mode-hook 'flyspell-mode)
(add-hook 'LaTeX-mode-hook 'LaTeX-math-mode)

(add-hook 'LaTeX-mode-hook 'turn-on-reftex)
(setq reftex-plug-into-AUCTeX t)

(setq LaTeX-indent-level 0)

(setq TeX-brace-indent-level 0)


(add-hook 'after-init-hook #'global-flycheck-mode)


;; Copy and paste to clipboard:
(defun copy-to-clipboard ()
  (interactive)
  (if (display-graphic-p)
      (progn
        (message "Yanked region to x-clipboard!")
        (call-interactively 'clipboard-kill-ring-save)
        )
    (if (region-active-p)
        (progn
          (shell-command-on-region (region-beginning) (region-end) "xsel -i -b")
          (message "Yanked region to clipboard!")
          (deactivate-mark))
      (message "No region active; can't yank to clipboard!")))
  )

(defun paste-from-clipboard ()
  (interactive)
  (if (display-graphic-p)
      (progn
        (clipboard-yank)
        (message "graphics active")
        )
    (insert (shell-command-to-string "xsel -o -b"))
    )
  )

(global-set-key [f8] 'copy-to-clipboard)
(global-set-key [f9] 'paste-from-clipboard)
