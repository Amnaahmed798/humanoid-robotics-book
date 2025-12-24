import { useState, useEffect } from 'react';

const useTextSelection = () => {
  const [selectedText, setSelectedText] = useState('');
  const [selectionContext, setSelectionContext] = useState({
    pageUrl: '',
    surroundingText: '',
    selectionStart: 0,
    selectionEnd: 0
  });

  useEffect(() => {
    const handleSelection = () => {
      const selected = window.getSelection().toString().trim();
      if (selected) {
        const range = window.getSelection().getRangeAt(0);
        const start = range.startOffset;
        const end = range.endOffset;

        // Get surrounding text (100 characters before and after)
        const textNode = range.startContainer;
        if (textNode.nodeType === Node.TEXT_NODE) {
          const text = textNode.textContent;
          const contextStart = Math.max(0, start - 100);
          const contextEnd = Math.min(text.length, end + 100);
          const surroundingText = text.substring(contextStart, contextEnd);

          setSelectedText(selected);
          setSelectionContext({
            pageUrl: window.location.href,
            surroundingText,
            selectionStart: start,
            selectionEnd: end
          });
        }
      } else {
        setSelectedText('');
        setSelectionContext({
          pageUrl: '',
          surroundingText: '',
          selectionStart: 0,
          selectionEnd: 0
        });
      }
    };

    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('keyup', handleSelection);

    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('keyup', handleSelection);
    };
  }, []);

  const getSelectedTextWithContext = () => {
    if (!selectedText) return null;

    return {
      text: selectedText,
      context: selectionContext
    };
  };

  return {
    selectedText,
    selectionContext,
    getSelectedTextWithContext,
    clearSelection: () => {
      setSelectedText('');
      setSelectionContext({
        pageUrl: '',
        surroundingText: '',
        selectionStart: 0,
        selectionEnd: 0
      });
    }
  };
};

export default useTextSelection;