import React from 'react';
import { Toaster, toast } from 'react-hot-toast';

export function notify(message, type = 'success') {
  if (type === 'success') toast.success(message);
  else if (type === 'error') toast.error(message);
  else toast(message);
}

function NotificationBar() {
  return <Toaster position="top-right" />;
}

export default NotificationBar; 